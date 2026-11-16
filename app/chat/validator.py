"""Plan validation against an allow-list built from the LOADED data. A plan that passes here can only
touch known, non-PII fields, declared joins, known operators and bounded limits.
Every rejection is a structured PipelineError with a stable code (the Step 8 repair loop feeds these to the LLM)."""
import difflib
import re
from dataclasses import dataclass, field
from typing import Union

import pandas as pd

from app.chat.catalog import is_pii, kind_of
from app.chat.plan import ClarifyPlan, QueryPlan, RefusePlan, RetailPlan
from app.config import settings
from app.errors import PipelineError
from app.retail.filters import OPS, apply_filters
from app.retail.inventory import INV_DIMENSIONS, INV_FILTER_FIELDS, INV_METRICS
from app.retail.metrics import DIMENSIONS, FILTER_FIELDS, METRICS
from app.retail.relationships import get_relationship

DEFAULT_LIMIT = 50
AGGS = {"count", "count_distinct", "sum", "avg", "min", "max", "median"}
_ALIAS = re.compile(r"^[a-z][a-z0-9_]{0,30}$")
# retail dimension -> dataset that must be loaded
DIM_NEEDS = {"region": "stores", "store": "stores", "city": "stores", "category": "products", "brand": "products", "product": "products",
             "customer_city": "customers", "store_id": None, "sku": None, "month": None, "status": None}


@dataclass
class Resolved:
    public: str
    column: str
    kind: str
    pii: bool
    dataset: str


@dataclass
class Validated:
    plan: Union[QueryPlan, RetailPlan, ClarifyPlan, RefusePlan]
    limit: int = DEFAULT_LIMIT
    fields: dict = field(default_factory=dict)          # public name -> Resolved (only fields the plan uses)
    right_columns: dict = field(default_factory=dict)   # joined dataset -> columns to bring in


def _check_limit(limit):
    if limit is None:
        return DEFAULT_LIMIT
    if limit < 1:
        raise PipelineError("INVALID_LIMIT", "limit must be at least 1.", {"limit": limit})
    if limit > settings.max_result_rows:
        raise PipelineError("LIMIT_TOO_LARGE", f"limit may not exceed {settings.max_result_rows}.", {"limit": limit, "max": settings.max_result_rows})
    return limit


def _check_dir(d):
    if d not in ("asc", "desc"):
        raise PipelineError("INVALID_SORT", "Sort direction must be 'asc' or 'desc'.", {"dir": str(d)[:10]})


def _check_ops(filters):
    for f in filters:
        if f.op not in OPS:
            raise PipelineError("INVALID_OPERATOR", "Unknown filter operator.", {"op": f.op, "allowed": sorted(OPS)})


# ------------------------------------------------------------------------------ generic query plans
def _resolver(data, base, joins):
    table = {}
    for c in data.frames[base].columns:
        r = Resolved(c, c, kind_of(data.frames[base][c]), is_pii(c), base)
        table[c] = table[f"{base}.{c}"] = r
    for j in joins:
        rel = get_relationship(base, j)
        for c in data.frames[j].columns:
            if c not in rel.right_keys:
                table[f"{j}.{c}"] = Resolved(f"{j}.{c}", f"{j}__{c}", kind_of(data.frames[j][c]), is_pii(c), j)

    def resolve(name):
        if name in table:
            return table[name]
        pre = name.split(".")[0] if "." in name else None
        if pre and pre in data.frames and pre != base and pre not in joins:
            raise PipelineError("FIELD_NOT_JOINED", f"'{name}' needs '{pre}' in joins.", {"field": name, "add_join": pre})
        public = sorted({r.public for r in table.values()})
        raise PipelineError("UNKNOWN_FIELD", "Unknown field.", {"field": name[:50], "suggestions": difflib.get_close_matches(name, public, n=3, cutoff=0.5),
                                                               "available": public[:40]})
    return resolve


def _validate_query(p: QueryPlan, data) -> Validated:
    if p.dataset not in data.frames:
        raise PipelineError("UNKNOWN_DATASET", "Unknown dataset.", {"dataset": p.dataset, "available": sorted(data.frames)})
    joins = list(dict.fromkeys(p.joins))
    for j in joins:
        get_relationship(p.dataset, j)                                   # JOIN_NOT_ALLOWED if not declared
        if j not in data.frames:
            raise PipelineError("DATASET_NOT_LOADED", f"Dataset '{j}' is not loaded.", {"dataset": j})
    resolve = _resolver(data, p.dataset, joins)
    used = {}

    def use(name, allow_pii=False):
        r = resolve(name)
        if r.pii and not allow_pii:
            raise PipelineError("FIELD_RESTRICTED", "Personal-data fields can only be counted, not shown, grouped or filtered.", {"field": r.public})
        used[r.public] = r
        return r

    _check_ops(p.filters)
    filters = [f.model_copy(update={"field": use(f.field).public}) for f in p.filters]
    limit = _check_limit(p.limit)
    for s in p.sort:
        _check_dir(s.dir)

    if p.intent == "aggregate":
        if not p.metrics:
            raise PipelineError("INVALID_PLAN", "aggregate needs at least one metric.")
        if p.select:
            raise PipelineError("INVALID_PLAN", "aggregate does not take 'select'.")
        group_by = [use(g).public for g in p.group_by]
        if len(set(group_by)) != len(group_by):
            raise PipelineError("INVALID_PLAN", "Duplicate group_by field.")
        metrics, aliases = [], set()
        for m in p.metrics:
            if m.agg not in AGGS:
                raise PipelineError("INVALID_AGGREGATION", "Unknown aggregation.", {"agg": m.agg, "allowed": sorted(AGGS)})
            if not _ALIAS.match(m.as_) or m.as_ in aliases or m.as_ in group_by:
                raise PipelineError("INVALID_ALIAS", "Metric aliases must be unique snake_case names that differ from group fields.", {"alias": m.as_})
            aliases.add(m.as_)
            if m.field is None:
                if m.agg != "count":
                    raise PipelineError("INVALID_AGGREGATION", f"'{m.agg}' needs a field.", {"agg": m.agg})
                metrics.append(m)
                continue
            r = use(m.field, allow_pii=m.agg in ("count", "count_distinct"))
            if m.agg in ("sum", "avg", "median") and r.kind != "number":
                raise PipelineError("INVALID_AGGREGATION", f"'{m.agg}' needs a numeric field.", {"field": r.public, "type": r.kind})
            if m.agg in ("min", "max") and r.kind == "text":
                raise PipelineError("INVALID_AGGREGATION", f"'{m.agg}' needs a numeric or date field.", {"field": r.public, "type": r.kind})
            metrics.append(m.model_copy(update={"field": r.public}))
        outputs = set(group_by) | aliases
        sort = []
        for s in p.sort:
            name = s.field if s.field in aliases else use(s.field).public
            if name not in outputs:
                raise PipelineError("INVALID_SORT", "Sort fields must be group_by fields or metric aliases.", {"field": s.field, "allowed": sorted(outputs)})
            sort.append(s.model_copy(update={"field": name}))
        norm = p.model_copy(update={"filters": filters, "group_by": group_by, "metrics": metrics, "sort": sort})
    elif p.intent == "select":
        if p.metrics or p.group_by:
            raise PipelineError("INVALID_PLAN", "select does not take metrics or group_by.")
        if not p.select:
            raise PipelineError("INVALID_PLAN", "select needs at least one field.")
        sel = [use(s).public for s in p.select]
        sort = [s.model_copy(update={"field": use(s.field).public}) for s in p.sort]
        norm = p.model_copy(update={"filters": filters, "select": sel, "sort": sort})
    elif p.intent == "distinct":
        if p.metrics or p.select:
            raise PipelineError("INVALID_PLAN", "distinct takes group_by fields only.")
        if not 1 <= len(p.group_by) <= 2:
            raise PipelineError("INVALID_PLAN", "distinct needs 1-2 fields in group_by.")
        fields = [use(g).public for g in p.group_by]
        sort = []
        for s in p.sort:
            name = s.field if s.field == "count" else use(s.field).public
            if name != "count" and name not in fields:
                raise PipelineError("INVALID_SORT", "Sort by one of the distinct fields or 'count'.", {"field": s.field})
            sort.append(s.model_copy(update={"field": name}))
        norm = p.model_copy(update={"filters": filters, "group_by": fields, "sort": sort})
    else:  # describe
        if p.metrics or p.group_by or p.sort:
            raise PipelineError("INVALID_PLAN", "describe takes only select fields (and filters).")
        if not 1 <= len(p.select) <= 6:
            raise PipelineError("INVALID_PLAN", "describe needs 1-6 fields in select.")
        norm = p.model_copy(update={"filters": filters, "select": [use(s).public for s in p.select]})

    # filter VALUES are type-checked now (against empty frames), so bad plans fail before any data is touched
    probe = data.frames[p.dataset].head(0)
    extra = {r.column: data.frames[r.dataset][r.column.split("__", 1)[1]].head(0) for r in used.values() if r.dataset != p.dataset}
    if extra:
        probe = pd.concat([probe.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    apply_filters(probe, [f.model_dump() for f in filters], {r.public: r.column for r in used.values()})

    right = {}
    for r in used.values():
        if r.dataset != p.dataset:
            right.setdefault(r.dataset, []).append(r.column.split("__", 1)[1])
    right = {k: sorted(set(v)) for k, v in right.items()}
    norm = norm.model_copy(update={"joins": [j for j in joins if j in right]})     # unused joins are dropped (less work)
    return Validated(norm, limit, used, right)


# ------------------------------------------------------------------------------ retail plans
def _validate_retail(p: RetailPlan, data) -> Validated:
    if p.domain not in ("sales", "inventory"):
        raise PipelineError("INVALID_PLAN", "domain must be 'sales' or 'inventory'.", {"domain": p.domain})
    sales = p.domain == "sales"
    need = "orders" if sales else "inventory"
    if need not in data.frames:
        raise PipelineError("DATASET_NOT_LOADED", f"'{p.domain}' metrics need the '{need}' dataset.", {"dataset": need})
    dims, mets, fmap = (DIMENSIONS, METRICS, FILTER_FIELDS) if sales else (INV_DIMENSIONS, INV_METRICS, INV_FILTER_FIELDS)
    for g in p.group_by:
        if g not in dims:
            raise PipelineError("INVALID_GROUP_BY", "Unknown dimension.", {"dimension": g, "allowed": sorted(dims)})
        n = DIM_NEEDS.get(g) if sales else {"region": "stores", "store": "stores", "city": "stores", "category": "products",
                                            "brand": "products", "product": "products"}.get(g)
        if n and n not in data.frames:
            raise PipelineError("DIMENSION_UNAVAILABLE", f"Dimension '{g}' needs the '{n}' dataset.", {"dimension": g, "needs": n})
    if len(set(p.group_by)) != len(p.group_by):
        raise PipelineError("INVALID_GROUP_BY", "Duplicate dimension.")
    for m in p.metrics:
        if m not in mets:
            raise PipelineError("INVALID_METRIC", "Unknown metric.", {"metric": m, "allowed": sorted(mets)})
    if p.sort_by and p.metrics and p.sort_by not in p.metrics:
        raise PipelineError("INVALID_SORT", "sort_by must be one of the requested metrics.", {"sort_by": p.sort_by})
    if p.sort_by and p.sort_by not in mets:
        raise PipelineError("INVALID_SORT", "Unknown sort metric.", {"sort_by": p.sort_by})
    _check_dir(p.direction)
    _check_ops(p.filters)
    if p.as_of is not None:
        if sales:
            raise PipelineError("INVALID_PLAN", "as_of only applies to inventory metrics.")
        try:
            pd.to_datetime(p.as_of, format="%Y-%m-%d")
        except (ValueError, TypeError):
            raise PipelineError("INVALID_AS_OF", "as_of must be an ISO date YYYY-MM-DD.", {"value": p.as_of[:10]})
    for f in p.filters:
        if f.field not in fmap:
            raise PipelineError("UNKNOWN_FIELD", "Unknown filter field.", {"field": f.field, "allowed": sorted(fmap)})
        n = DIM_NEEDS.get(f.field) if sales else None
        if n and n not in data.frames:
            raise PipelineError("DIMENSION_UNAVAILABLE", f"Field '{f.field}' needs the '{n}' dataset.", {"field": f.field, "needs": n})
    return Validated(p, _check_limit(p.limit))


def validate_plan(plan, data) -> Validated:
    if isinstance(plan, QueryPlan):
        return _validate_query(plan, data)
    if isinstance(plan, RetailPlan):
        return _validate_retail(plan, data)
    return Validated(plan)           # clarify / unanswerable carry no executable content
