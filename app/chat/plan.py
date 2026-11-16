"""Structured query plans. The LLM (or a human, or a test) may only EXPRESS a plan in this schema;
nothing here executes anything. Unknown keys, wrong types, over-long strings and oversized lists are rejected.
Semantic checks (does the field exist? is the join allowed?) live in validator.py."""
import json
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, StringConstraints, TypeAdapter, ValidationError

from app.errors import PipelineError

MAX_PLAN_CHARS = 20_000
FieldName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,40}(\.[A-Za-z_][A-Za-z0-9_]{0,40})?$")]
Ident = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,49}$")]
Word = Annotated[str, StringConstraints(pattern=r"^[a-z_]{1,20}$")]
Scalar = Union[StrictStr, StrictInt, StrictFloat, StrictBool, None]
Value = Union[Scalar, Annotated[list[Union[StrictStr, StrictInt, StrictFloat]], Field(max_length=100)]]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_max_length=500)


class FilterSpec(_Strict):
    field: FieldName
    op: Word
    value: Value = None


class MetricSpec(_Strict):
    model_config = ConfigDict(extra="forbid", str_max_length=500, populate_by_name=False)
    agg: Word
    field: Optional[FieldName] = None
    as_: Ident = Field(alias="as")


class SortSpec(_Strict):
    field: FieldName
    dir: Word = "asc"


class QueryPlan(_Strict):
    """Generic table operations over ONE dataset, optionally joined through declared relationships."""
    intent: Literal["aggregate", "select", "distinct", "describe"]
    dataset: Ident
    joins: list[Ident] = Field(default_factory=list, max_length=3)
    filters: list[FilterSpec] = Field(default_factory=list, max_length=10)
    group_by: list[FieldName] = Field(default_factory=list, max_length=3)
    metrics: list[MetricSpec] = Field(default_factory=list, max_length=6)
    select: list[FieldName] = Field(default_factory=list, max_length=12)
    sort: list[SortSpec] = Field(default_factory=list, max_length=3)
    limit: Optional[StrictInt] = None
    inherits_filters: bool = False      # True = also apply the previous turn's filters (merged by code, see planner.apply_inheritance)


class RetailPlan(_Strict):
    """Named, auditable business metrics (revenue, AOV, return rate, stock-out, sell-through...)."""
    intent: Literal["retail_metric"]
    domain: Word
    group_by: list[Ident] = Field(default_factory=list, max_length=3)
    metrics: list[Ident] = Field(default_factory=list, max_length=13)
    filters: list[FilterSpec] = Field(default_factory=list, max_length=10)
    sort_by: Optional[Ident] = None
    direction: Word = "desc"
    limit: Optional[StrictInt] = None
    as_of: Optional[Annotated[str, StringConstraints(max_length=10)]] = None
    inherits_filters: bool = False


class ClarifyPlan(_Strict):
    intent: Literal["clarify"]
    question: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    options: list[Annotated[str, StringConstraints(max_length=60)]] = Field(default_factory=list, max_length=5)


class RefusePlan(_Strict):
    intent: Literal["unanswerable"]
    reason: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    missing: list[Annotated[str, StringConstraints(max_length=80)]] = Field(default_factory=list, max_length=10)


Plan = Annotated[Union[QueryPlan, RetailPlan, ClarifyPlan, RefusePlan], Field(discriminator="intent")]
_ADAPTER = TypeAdapter(Plan)


def parse_plan(raw) -> Union[QueryPlan, RetailPlan, ClarifyPlan, RefusePlan]:
    """raw: dict or JSON string. Never echoes the offending input back (it may be attacker-controlled)."""
    if isinstance(raw, (str, bytes)):
        if len(raw) > MAX_PLAN_CHARS:
            raise PipelineError("INVALID_PLAN", "Plan is too large.", {"max_chars": MAX_PLAN_CHARS})
        try:
            raw = json.loads(raw)
        except (ValueError, RecursionError):
            raise PipelineError("INVALID_PLAN", "Plan is not valid JSON.")
    if not isinstance(raw, dict):
        raise PipelineError("INVALID_PLAN", "Plan must be a JSON object.")
    try:
        return _ADAPTER.validate_python(raw, by_alias=True)
    except ValidationError as e:
        errs = [{"loc": ".".join(str(x) for x in er["loc"])[:80], "type": er["type"], "msg": er["msg"][:100]} for er in e.errors()[:8]]
        raise PipelineError("INVALID_PLAN", "Plan does not match the schema.", {"errors": errs})
