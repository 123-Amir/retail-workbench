# AI Retail Data & Conversational Intelligence Workbench

This project cleans messy retail CSV files and lets users ask questions about the cleaned data.

The application, not the model, calculates every number:

- The pipeline profiles, cleans and validates data.
- The retail layer calculates sales and inventory metrics.
- The chat layer turns questions into safe JSON plans.
- The executor validates and runs those plans.

The default provider is a deterministic, keyless `mock` provider. No paid service is needed for the tests or evaluator.

## 1. Quick start

Use Python **3.11+**. The project was tested with Python 3.12.

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python -m pytest -q
python -m app.evaluate --input cases.json --output results.json --artifacts-dir ./evaluation_output
uvicorn app.api.main:create_app --factory --port 8000
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env

.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m app.evaluate --input cases.json --output results.json --artifacts-dir .\evaluation_output
.\.venv\Scripts\python.exe -m uvicorn app.api.main:create_app --factory --port 8000
```

This is an **API/CLI project**. It has no separate frontend. Use the FastAPI routes, walkthrough, and demo scripts to try it.

Useful commands:

- `python scripts/walkthrough.py [1-5]` — run one walkthrough step.
- `python scripts/generate_sample_data.py` — regenerate sample data.
- `python scripts/clean_demo.py`, `retail_demo.py`, or `chat_demo.py` — run a smaller demo.

### Providers and authentication

- `LLM_PROVIDER=mock` is the default.
- `LLM_PROVIDER=anthropic` uses the Anthropic adapter.
- Set `ANTHROPIC_API_KEY` and optionally `ANTHROPIC_MODEL` for Anthropic.
- Live Anthropic behavior is not verified in this repository.
- `API_TOKEN` enables bearer-token authentication.
- `X-Owner-Id` selects the data owner and isolation scope.
- Without `API_TOKEN`, `X-Owner-Id` is not standalone authentication. Put the API behind an authenticating proxy.

For a quick review, run the evaluator and open the cleaned CSV, report, and `chat_trace.json` artifacts in `evaluation_output/`.

## 2. Architecture

```
            ┌────────────────────────── HTTP API (FastAPI) / evaluator CLI ──────────────────────────┐
            │                                   same production services                              │
            ▼                                                                                          ▼
   RunService (storage/runs.py)                                                          ChatService (chat/service.py)
   submit → background job → status                                                      sessions · turns · caches
   idempotent · isolated per owner                                                              │
            │                                                                                    ▼
            ▼                                                                    Planner (chat/planner.py)
   pipeline/  loader → profiler → issues → planner → cleaner → validate                 prompt ◄── catalog (bounded, PII-free)
   (pure, deterministic, no LLM)                    │                                    LLM  ──► JSON plan ──┐ repair ≤3 calls
            │                                       ▼                                    retry/backoff·breaker │ fallback: rules.py
            ▼                                 artifacts on disk                                               ▼
   SQLite (runs, datasets, events,             original · cleaned · report            validator (allow-list) → executor → evidence
   chat_sessions, chat_turns)                  sha256 + data_version                          │               retail/ joins · filters
                                                                                              ▼               metrics · inventory
                                                                              answer text = template over the RESULT;
                                                                              optional LLM wording must pass numeric grounding
```

| Package | Responsibility |
|---|---|
| `app/pipeline` | `loader` (safe CSV, encodings), `parsers`, `profiler`, `issues` (16 issue types), `variants`, `planner` (cleaning plan), `cleaner` (pure executors), `validate` (independent checks) |
| `app/storage` | SQLite state, atomic hashed artifacts, `RunService` (queued→running→succeeded/partial/failed), typed re-load of cleaned data |
| `app/retail` | Declared relationships, safe joins, strict filters, sales and inventory metrics with explicit definitions |
| `app/chat` | Plan schema (Pydantic), allow-list validator, executor, planner, rule-based fallback, grounding, sessions |
| `app/llm` | Provider interface, Anthropic adapter, mock/scripted providers, retry + circuit breaker |
| `app/evaluate` | Batch evaluator, case parser, expectation engine |
| `app/api` | Thin HTTP layer (no business logic) |

**Data flow.** Upload → `RunService.submit` (type/size/name checks, SHA-256, original saved byte-for-byte, idempotency key) → background
job per dataset: load as *strings* → profile → issues → cleaning plan → execute → profile again → independent validation → write cleaned CSV
atomically → **read it back and compare** → publish `data_version = name@sha256[:12]`. A dataset whose validation fails publishes **no** cleaned artifact.
Chat sessions are pinned to one run, so every answer cites the data versions it used.

## 3. The AI-vs-code boundary

| Concern | Owner | How AI output is controlled |
|---|---|---|
| Counts, joins, filters, aggregates, schema checks, duplicate detection, scoring arithmetic | **Code only** | n/a — the model never sees rows and never produces a number that is shown |
| Cleaning plan and every transformation | **Code** (`source: "code"` on every step) | AI-suggested steps are *not implemented*; the `source` field exists in the schema for them |
| Natural language → query plan | **LLM** (or the rule planner) | Strict Pydantic schema → allow-list validator → bounded executor; rejected plans trigger a ≤3-call repair loop with the validator's structured error |
| Follow-up state ("only West", "among those…") | **Code** (`inherits_filters` merge) | The model only says *that* it is a follow-up; code merges filters |
| Wording of the answer | Template (default) or **LLM** (optional) | Every number in LLM wording must already exist in the result/evidence/question, otherwise it is discarded |
| Refusing / clarifying | Plan types `clarify` and `unanswerable` | First-class plans; nothing executes |

## 4. Data pipeline

* **Profile** (before and after): rows/columns, nulls (empty vs placeholder like `N/A`), duplicates (exact and after normalisation), cardinality,
  inferred type with parse failures, numeric percentiles, date formats and ambiguity counts, case/whitespace variant groups, top values (PII redacted).
* **16 issue types**, each with a stable id (`sha1(dataset|type|columns)`, independent of row order): duplicate rows, whitespace, placeholder nulls, SKU format,
  category/brand/region variants, mixed numeric formats (`$1,299.00`, `Rs 499`, `10%`), parse failures, mixed date formats, **ambiguous dates**, negative quantities,
  all-null columns, key conflicts, prompt-injection-shaped cell text, column-name format, empty dataset, duplicates created by cleaning.
* **Cleaning plan before any mutation**: per step `id, reason, fields, risk, source, status (planned | needs_review | flag_only), default_action, params, expected_effect`;
  after execution `final_status` (applied | skipped | failed | flagged) and an `execution` log (cells changed/nulled, rows removed, examples, row numbers).
  No step can drop a column; rows are removed only by exact-duplicate removal and the count is verified.
* **Ambiguous dates are never guessed.** `03/04/2025` is resolved only from *unambiguous* dates with the same separator (≥5 observations, ≥95 % agreement);
  otherwise the value becomes null and is reported (`values_nulled_by_reason`).
* **Independent validation** (recomputed from the real before/after frames, not from executor logs): schema preserved, row loss fully explained, no duplicates left,
  expected effect met, every new null explained by a logged step, no failed steps, no actionable issues left, **idempotency** (re-cleaning the output changes nothing).
* **Trade-off made visible:** if cleaning makes two *different* raw rows identical (e.g. two unparseable dates nulled), they are merged so that the output is a fixed point,
  and the merge is reported as a `warn` (`no_rows_merged_by_data_loss`) with row numbers. See section 12.

## 5. Retail-domain intelligence (3 of the 4 offered capabilities)

1. **Product / category normalisation** — case/whitespace/plural/punctuation clusters (`T-Shirt`, `tshirt`, `t-shirts`) with confidence 1.0/0.95; curated alias tables
   (`Tee→T-Shirt`, `Denim→Jeans`, `W→West`) at confidence 0.8 are flagged `needs_review`. SKUs canonicalised to `PREFIX-0000`. Every mapping carries its method and confidence.
2. **Cross-dataset relationships** — only declared many-to-one joins (`orders→products|stores|customers`, `inventory→products|stores`). Joins are positional lookups,
   so **fan-out is impossible by construction**; duplicate dimension keys raise `JOIN_FANOUT`; null keys never match each other; the output row count/order is asserted;
   every join reports match rate, orphan keys (e.g. `SKU-9999`: 31 lines) and unreferenced dimension keys. Store×SKU many-to-many is pre-aggregated before joining.
3. **Sales / inventory analytics** — 13 sales metrics (gross/net revenue, orders, AOV, return rate by units/value, cancellation rate, margin, cost coverage…) and
   11 inventory metrics (stock-out, low stock, sell-through, ageing…). Definitions are in the `app/retail/metrics.py` and `inventory.py` docstrings and are returned with every
   answer (`evidence.formulas`). Invalid lines are **excluded and counted, never repaired**; orphan SKUs/stores stay visible as `(unknown)`; each answer lists `missing_evidence`
   (e.g. "stock is a snapshot without an as-of timestamp, so stock-out is a point-in-time share, not a frequency").

Not implemented: the optional *performance / recommendation scoring* capability.

## 6. Chat assistant

* **Plans** (`app/chat/plan.py`): `retail_metric`, and `aggregate | select | distinct | describe` on raw columns with declared joins, plus `clarify` and `unanswerable`.
  Unknown keys, wrong types, injection-shaped field names and oversized lists are rejected by the schema.
* **Allow-list validator** (`validator.py`): known datasets/fields only, declared joins only, personal-data fields can only be *counted*, operator and aggregation allow-lists,
  `limit ≤ 200` (rejected, not clamped), ≤3 joins/group-bys, ≤10 filters, filter values type-checked against empty frames before execution. Errors have stable codes
  (`UNKNOWN_FIELD` with suggestions, `JOIN_NOT_ALLOWED`, `FIELD_RESTRICTED`, `LIMIT_TOO_LARGE`, …).
* **Executor**: no dynamic code or SQL; row-scan cap, wall-clock timeout (soft: a Python thread cannot be killed), deterministic ordering.
* **Evidence on every answer**: `datasets` (with `data_version`), `columns_used`, `filters` (rows before/after), `joins` (reports), `operations`, `rows_considered/rows_total`,
  `preview_limit`, plus `exclusions`, `formulas`, `missing_evidence` for retail metrics. A contract test checks this for all six answer types.
* **Multi-turn**: "net revenue by region" → "only stores in the West region" → "among those, which categories had the highest return rate?" keeps the West filter (380 rows)
  and re-groups; a new filter on the same field replaces the old one; error turns do not damage state.
* **Sessions** persist `question → raw plan → validated plan → planner attempts → execution trace + evidence + result preview → answer`. System prompts, catalogs sent to the model,
  keys and unparseable model output are never stored.
* **Challenge cases**: ambiguous (`clarify`), unanswerable and protected-trait questions (`unanswerable`), prompt injection in the question or in a cell, malicious/expensive plans,
  LLM outage or invalid JSON (retry → repair → deterministic fallback), raw-vs-cleaned (answers always cite the cleaned `data_version`; asking for "raw" gets a clarification).
  Each has a sample evaluation case and a unit test.

## 7. Provider, model and what leaves the system

Providers: `mock` (default, deterministic rule-based, no network), `anthropic` (Messages API, `temperature=0`, model from `ANTHROPIC_MODEL`), `scripted` (tests / evaluator `model_script`).

**Sent to an external model** (planning call): the user's question; the previous question and previous *plan*; dataset names (derived from file names) and row counts; column
names/types and a PII flag; distinct *labels* of low-cardinality, non-PII, non-identifier text columns (≤15 per column, ≤40 chars, control characters stripped, injection-shaped labels removed);
declared joins; metric/dimension definitions. On a repair call additionally: the validator's error and the model's own previous plan.
**Optional explanation call** (aggregate / distinct / retail results only): the question, column names, ≤20 result rows (strings truncated, injection-shaped cells replaced by `[redacted]`),
totals, formulas and ≤5 caveats.
**Never sent**: raw rows, `select`/`describe` results, PII column values, identifier values (`order_id`, `sku`, `customer_id`, `store_id`), file contents, owner ids, secrets.

## 8. Security and privacy decisions

* Filenames, CSV cells, model output and plans are untrusted: filename sanitisation, `.csv` only, size cap (`MAX_UPLOAD_MB`), `safe_join` path containment, owner-id regex.
* Isolation: every service call is scoped by owner; another owner's run/session returns the same error as a missing one (no existence leak); files live under `<root>/<owner>/<run>/`.
* PII (`email`, `name`, phone, address…) is redacted from profiles, examples and top-values, cannot be shown/grouped/filtered in chat, and is never sent to a model.
* Prompt injection: defence is architectural (plans are data, validated against an allow-list, executed by fixed code); regex detection only *flags* suspicious cells and strips them from prompts.
* Model output can never bypass authorisation: the model does not know owner ids and cannot name a dataset the owner has not loaded.
* Secrets only via environment (`.env` is git-ignored); errors never contain stack traces or internal paths (tested).
* Protected traits are not analysed or inferred: refused by the rule planner and excluded by design (no demographic columns; PII fields restricted). With a real model this relies on the prompt plus those structural limits.

## 9. Evaluation

```bash
python -m app.evaluate --input cases.json --output results.json --artifacts-dir ./evaluation_output [--provider mock|anthropic] [--strict] [--store-dir DIR]
```

* One failing case, file or question never aborts the batch (structured error, run continues). Exit code 0 when the batch completed; `--strict` exits 1 if any case failed;
  2 = fatal input/config problem (output file still written with `fatal_error`).
* **Case format** (`cases.json` = `cases/sample_cases.json`): `{"cases": [{"id", "files": [path | {"path","name"}], "decisions": {step_id: "apply"|"skip"}, "model_script": [...], "expect": {...}, "questions": [{"question"|"plan", "expect": {...}}]}]}`.
  Paths are resolved relative to the cases file, then the working directory. `{"path": "x.csv", "name": "orders"}` maps a differently named file onto a canonical dataset name.
* **Expectations**: dataset (`status, rows_in/out/removed, validation_status, idempotent, issue_types_include/exclude, plan_actions_include, columns, cleaned_columns{all_match,unique_count,null_count,values_in}`),
  question (`status, error_code, plan_intent, plan_subset, planner_source, row_count, rows, rows_unordered, answer_contains, numbers_grounded, evidence_has, assertions[]`).
  Assertion operators: `equals approx contains not_contains gte lte gt lt in is_null exists length regex sorted_by`.
* **Output** (`results.json`): `{"schema_version", "summary", "results": [...]}`. Each result has the required keys `dataset, profile_before, issues, cleaning_plan, profile_after, validation, chat_evaluation[{question, plan, answer, trace, status}], artifacts`
  plus `case_id, status (passed|failed|error), checks, error, datasets`. Single-file cases are flat; multi-file cases are keyed by dataset name in `dataset/profile_*/validation`; `datasets` is always keyed.
* **Deterministic**: no timestamps, run/session ids, temp paths or durations are written; two runs produce byte-identical `results.json` and artifacts (also across pandas 3.0.6 and 3.1.1).
* **Artifacts** per case: `<dataset>.original.csv`, `<dataset>.cleaned.csv`, `<dataset>.report.json`, `chat_trace.json`.
* The shipped 15 sample cases cover cleaning, edge files (header-only, empty, malformed, cp1252, duplicate columns, all-null, ambiguous dates, key conflict), multi-dataset retail chat,
  hostile plans, prompt injection, and scripted model failures. Expected numbers in them are golden values from this engine, cross-checked by independent pandas calculations in the unit tests.

## 10. HTTP API

Run one process: `uvicorn app.api.main:create_app --factory --port 8000`. Identity via `X-Owner-Id` (put behind an authenticating proxy) and optional `API_TOKEN` bearer token.

| Method & path | Purpose |
|---|---|
| `POST /runs` (multipart `files`, optional `decisions` JSON) | Upload; returns 202 + `run_id`; processed in the background; identical content → same run (`reused: true`) |
| `GET /runs`, `GET /runs/{id}` | List / status per dataset (no internal paths) |
| `GET /runs/{id}/datasets/{name}/report` | Profiles, issues, plan with execution logs, validation |
| `GET /runs/{id}/datasets/{name}/cleaned` | Hash-verified cleaned CSV |
| `POST /runs/{id}/sessions`, `GET /sessions`, `GET /sessions/{id}` | Chat sessions and full replay |
| `POST /sessions/{id}/ask` `{"question": ...}` or `{"plan": {...}}` | One turn (analytic failures are `200` with `status: "error"`; request problems are 4xx) |

## 11. Tests

373 automated tests (`pytest -q`). The ones that matter most: profiler null accounting and determinism; issue/step id stability under row shuffling; **cleaner idempotency and "validator catches sabotage"**
(silent null, row loss, schema mutation are injected and must fail validation); storage idempotency under 8 concurrent submits, tamper detection, crash recovery, owner isolation;
retail numbers checked by hand on an 8-line dataset and independently with pandas on the samples, partition invariants (groups sum to the total), fan-out refusal; validator rejection matrix;
multi-turn state; model-failure paths (outage, invalid JSON, hostile output, invented numbers); evaluator isolation and byte-for-byte determinism; 120k-row end-to-end; API isolation and error hygiene.

## 12. Trade-offs and known limitations

**Design trade-offs**
* *Idempotency vs. information:* merging rows that became identical only because a value was nulled keeps the cleaner a fixed point but loses a distinction. It is counted, row-numbered and `warn`ed rather than hidden.
* *Positional joins* trade a little generality for the guarantee that rows can never multiply.
* *Follow-up state in code* trades some model flexibility for reproducibility and testability.
* *Strings first:* everything is loaded as text so the profiler (not pandas) decides types; typed frames are restored from the recorded schema for analytics.
* *SQLite + filesystem + in-process thread pool* keeps setup to one `pip install`; it is not a distributed design.

