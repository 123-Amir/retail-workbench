"""Deterministic question -> plan translator. Two jobs:
  1. fallback planner when the LLM is down, rate limited or returns unusable output (data questions stay answerable)
  2. the brain of the 'mock' LLM provider, so the whole system is testable and evaluable without a paid service
It only EXPRESSES plans in the same schema the real model uses; the validator and executor treat it identically.
It prefers to ask or refuse over guessing."""
import re

from app.pipeline.variants import norm_key

UNANSWERABLE = [
    (("supplier", "vendor", "lead time"), ["supplier data"], "Supplier information is not part of the loaded datasets."),
    (("competitor", "market share"), ["competitor data"], "Competitor or market data is not part of the loaded datasets."),
    (("weather", "temperature", "holiday"), ["external weather/calendar data"], "External data such as weather is not available."),
    (("forecast", "predict", "projection", "next month", "next quarter", "next year", "will sell"), ["forecasting"],
     "Only historical data can be analysed; forecasting is not supported."),
    (("website", "traffic", "conversion rate", "footfall", "page view"), ["traffic / conversion data"], "No traffic or visit data is loaded, so conversion cannot be computed."),
]
PROTECTED = ("gender", "religion", "ethnic", "race of", "caste", "sexual", "disabilit", "customer age", "age group", "how old", "political")

SALES_METRICS = [("average order value", "aov"), ("avg order value", "aov"), ("aov", "aov"), ("net revenue", "net_revenue"),
                 ("gross sales", "gross_sales"), ("returned sales", "returned_sales"), ("revenue", "net_revenue"), ("sales", "net_revenue"),
                 ("return rate", "return_rate_units"), ("returns rate", "return_rate_units"), ("cancellation", "cancellation_rate"),
                 ("margin", "margin_pct"), ("profit", "gross_margin"), ("how many orders", "orders"), ("number of orders", "orders"),
                 ("order count", "orders"), ("orders", "orders"), ("units sold", "units_shipped"), ("units", "units_shipped")]
INV_METRICS = [("stock-out", "stock_out_rate"), ("stock out", "stock_out_rate"), ("stockout", "stock_out_rate"), ("out of stock", "stock_out_rate"),
               ("sell-through", "sell_through"), ("sell through", "sell_through"), ("low stock", "low_stock_positions"),
               ("units on hand", "units_on_hand"), ("on hand", "units_on_hand"), ("ageing", "avg_age_days"), ("aging", "avg_age_days"),
               ("aged", "aged_90_plus_positions"), ("restock", "avg_age_days")]
DIMS = {"region": "region", "regions": "region", "category": "category", "categories": "category", "store": "store", "stores": "store",
        "brand": "brand", "brands": "brand", "month": "month", "months": "month", "monthly": "month", "city": "city", "cities": "city",
        "product": "product", "products": "product", "sku": "sku", "skus": "sku"}
STOP = {"for", "in", "where", "with", "among", "only", "from", "of", "during", "at", "when", "that", "which", "had", "has", "have"}
FILTER_COLUMNS = {("stores", "region"): "region", ("stores", "store_name"): "store", ("stores", "city"): "city",
                  ("products", "category"): "category", ("products", "brand"): "brand", ("orders", "status"): "status"}
FOLLOWUP = ("among those", "of those", "those stores", "those categories", "those products", "from those", "same ", "instead", "what about",
            "how about", "and for ", "now ", "only ", "just ", "filter ", "narrow")
HIGH, LOW = ("highest", "most", "best", "top", "largest", "biggest", "worst return", "max"), ("lowest", "least", "worst", "bottom", "fewest", "smallest", "min")
RAW_WORDS = ("raw data", "raw file", "original data", "uncleaned", "before cleaning", "unclean", "raw version")


def _clarify(question, options=()):
    return {"intent": "clarify", "question": question[:300], "options": [o[:60] for o in options][:5]}


def _find_metrics(q, table):
    found = []
    for phrase, metric in sorted(table, key=lambda x: -len(x[0])):
        i = q.find(phrase)
        if i >= 0 and not any(i < e and i + len(phrase) > s for s, e, _ in found):
            found.append((i, i + len(phrase), metric))
    ordered = []
    for _, _, m in sorted(found):
        if m not in ordered:
            ordered.append(m)
    return ordered


def _find_dims(tokens):
    dims = []
    for i, tok in enumerate(tokens):
        if tok in ("by", "per", "each", "across") or (tok in ("which", "what") and i + 1 < len(tokens)):
            for t in tokens[i + 1:i + 6]:
                if t in STOP:
                    break
                if t in DIMS and DIMS[t] not in dims:
                    dims.append(DIMS[t])
                if t == "which":
                    break
    return dims[:3]


def _find_filters(q_tokens, catalog, domain):
    grams = q_tokens + [" ".join(q_tokens[i:i + 2]) for i in range(len(q_tokens) - 1)]
    keys = {norm_key(g): g for g in grams if len(g) > 2}
    filters = []
    for (ds, col), field in FILTER_COLUMNS.items():
        if field == "status" and domain == "inventory":
            continue
        meta = catalog["datasets"].get(ds)
        cd = next((c for c in meta["columns"] if c["name"] == col), None) if meta else None
        for v in (cd or {}).get("distinct_values", []):
            if norm_key(v) in keys and not any(f["field"] == field for f in filters):
                filters.append({"field": field, "op": "eq", "value": v})
    return filters


def rule_plan(question: str, catalog: dict, previous_plan: dict | None = None) -> dict:
    q = " ".join(question.casefold().replace("?", " ").replace(",", " ").split())
    tokens = re.findall(r"[a-z0-9\-']+", q)

    for words, missing, reason in UNANSWERABLE:
        if any(w in q for w in words):
            return {"intent": "unanswerable", "reason": reason, "missing": missing}
    if any(w in q for w in PROTECTED):
        return {"intent": "unanswerable", "reason": "I do not analyse or infer protected personal traits; only explicit retail signals are used.",
                "missing": ["protected-attribute analysis (not supported by design)"]}
    if any(w in q for w in RAW_WORDS):
        return _clarify("Chat runs on the cleaned dataset version (the raw upload is archived but not queryable). Continue with the cleaned data?",
                        ["Yes, use cleaned data"])

    if re.search(r"how many (?:rows|records|lines|entries)", q):
        ds = next((d for d in catalog["datasets"] if d in tokens or d.rstrip("s") in tokens), None)
        if not ds:
            return _clarify("Which dataset should I count rows in?", sorted(catalog["datasets"]))
        return {"intent": "aggregate", "dataset": ds, "metrics": [{"agg": "count", "as": "row_count"}]}

    inv = _find_metrics(q, INV_METRICS)
    sales = _find_metrics(q, SALES_METRICS)
    domain = "inventory" if inv else "sales"
    metrics = inv if inv else sales
    dims = _find_dims(tokens)
    filters = _find_filters(tokens, catalog, domain)
    followup = bool(previous_plan and previous_plan.get("intent") == "retail_metric"
                    and (any(m in q + " " for m in FOLLOWUP) or (not metrics and not dims and filters)))

    n = re.search(r"\b(?:top|bottom|first|lowest|highest)\s+(\d{1,3})\b", q)
    limit = int(n.group(1)) if n else None
    direction = "asc" if any(w in q for w in LOW) and not any(w in q for w in ("top ", "highest", "best")) else "desc"

    if followup:
        prev = previous_plan
        if metrics and (domain != prev["domain"]):
            return _build(domain, metrics, dims, filters, limit, direction, inherits=True)
        return _build(prev["domain"], metrics or prev["metrics"], dims if dims else prev["group_by"], filters, limit, direction, inherits=True)

    ranking = any(w in q for w in ("best", "top ", "worst", "bottom", "performing", "highest", "lowest"))
    if not metrics:
        if dims or ranking:
            return _clarify("Which measure should I use?", ["net revenue", "orders", "return rate", "average order value"])
        ds = next((d for d in catalog["datasets"] if d in tokens or d.rstrip("s") in tokens), None)
        if any(w in q for w in ("distinct", "unique", "different", "what values")) or q.startswith("which values"):
            hit = _match_column(q, tokens, catalog, ds)
            if hit:
                return {"intent": "distinct", "dataset": hit[0], "group_by": [hit[1]]}
        if any(w in q for w in ("describe", "summary", "statistics", "stats", "distribution", "summarize")):
            hit = _match_column(q, tokens, catalog, ds)
            if hit:
                return {"intent": "describe", "dataset": hit[0], "select": [hit[1]]}
        return _clarify("I could not map that to a supported analysis. Try a measure such as net revenue, orders, return rate, margin, stock-out rate "
                        "or sell-through, optionally by region, category, store or month.", ["net revenue by region", "return rate by category", "stock-out rate by region"])
    return _build(domain, metrics, dims, filters, limit, direction, inherits=False)


def _build(domain, metrics, dims, filters, limit, direction, inherits):
    plan = {"intent": "retail_metric", "domain": domain, "group_by": list(dims), "metrics": list(metrics), "filters": filters,
            "sort_by": metrics[0] if metrics else None, "direction": direction, "inherits_filters": inherits}
    if limit:
        plan["limit"] = limit
    return plan


def _match_column(q, tokens, catalog, preferred_ds):
    best = None
    for ds, meta in catalog["datasets"].items():
        for c in meta["columns"]:
            if c["pii"]:
                continue
            words = c["name"].replace("_", " ")
            if words in q or c["name"] in tokens or words + "s" in q:
                score = (ds == preferred_ds, len(c["name"]))
                if best is None or score > best[0]:
                    best = (score, ds, c["name"])
    return (best[1], best[2]) if best else None
