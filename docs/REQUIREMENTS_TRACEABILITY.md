# Requirements traceability

Status meanings:

- **Done:** implemented and covered by tests or evaluator cases.
- **Partial:** implemented with a documented limit.
- **Not done:** not implemented.

Tests are in `tests/`. `eval:` refers to cases in `cases.json`.

## Data and retail

| Requirement | Status | Evidence |
|---|---|---|
| Upload CSVs and persist original/cleaned artifacts | Done | `storage/runs.py`, `test_storage.py` |
| Profile before and after cleaning | Done | `pipeline/profiler.py`, `test_profiler.py` |
| Detect types, nulls, duplicates, variants, dates, and numbers | Done | `pipeline/issues.py`, `test_issues_plan.py` |
| Plan before mutation and record execution | Done / Partial | `pipeline/planner.py`; AI-suggested steps are not implemented |
| Validate schema, row loss, nulls, unresolved issues, and idempotency | Done | `pipeline/cleaner.py`, `pipeline/validate.py`, `test_cleaner.py` |
| Retail capabilities | Done (3 of 4) | `retail/variants.py`, `retail/joins.py`, `retail/metrics.py`, `retail/inventory.py` |
| Safe declared joins and fan-out checks | Done | `test_retail.py` |
| Auditable metrics with formulas and caveats | Done | Retail evidence and `test_retail.py` |
| Async processing, idempotency, caching, and lineage | Done | `storage/runs.py`, `test_storage.py` |
| Bounded memory | Partial | Upload cap and bounded previews; files are still read whole |

## Chat assistant

| Requirement | Status | Evidence |
|---|---|---|
| Filtering, sorting, grouping, aggregation, distinct, and statistics | Done | `chat/plan.py`, `test_chat_core.py` |
| Multi-turn context | Done | `test_chat_service.py` |
| Validated joins | Done | Declared relationships and `test_retail.py` |
| Evidence-backed answers | Done | Evidence contract tests and persisted chat traces |
| Grounded numerical claims | Done | `chat/grounding.py` |
| Clarification and refusal | Done | Chat service tests and evaluator cases |
| Prompt-injection handling | Done | Chat tests and `chat-prompt-injection-in-cell` |
| Malicious-plan rejection | Done | Validator tests and `chat-validation-and-safety` |
| LLM retry, fallback, and invalid-output handling | Done | `llm/resilience.py`, `test_llm_planner.py` |
| Raw versus cleaned selection | Done | Sessions are pinned to cleaned data versions |
| Customer/product segmentation | Partial | Product and customer dimensions exist; no dedicated segmentation workflow |

## Evaluation, security, and AI engineering

| Requirement | Status | Evidence |
|---|---|---|
| Batch evaluator uses production services | Done | `app/evaluate`, `test_evaluate.py` |
| One bad case does not stop the batch | Done | `test_one_bad_case_never_aborts_the_batch` |
| Deterministic output and traceable artifacts | Done | Evaluator determinism tests |
| Edge cases and model failures | Done | `edge-*`, `model-*`, and `test_llm_planner.py` |
| Untrusted filenames, cells, plans, and model output | Done | Loader, validator, path checks, isolation tests |
| PII minimization | Done | Catalog tests and prompt-sanitization tests |
| Safe structured plan execution | Done | Pydantic plans, allow-lists, fixed executors |
| Hard resource timeout | Partial | Timeout is a soft Python-thread deadline |
| Live-model quality benchmark | Partial | Mock/scripted evaluation exists; no live benchmark |
| Agent framework | Partial | Bounded plan, repair, retry, and fallback loop only |

## Submission notes

- The project is API/CLI based and has no frontend.
- The default provider is the keyless `mock` provider.
- Live Anthropic behavior is not verified.
- Authentication needs an authenticating proxy unless `API_TOKEN` is configured.
- Recommendation/scoring, column aliases, streaming cleaning, and run retention are not implemented.
