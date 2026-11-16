# Final Audit Report

## 1. Overall assessment

**Revised score: 88 / 100**

The project is substantially implemented and demonstrates a strong deterministic data-cleaning and safe-plan execution design. The confirmed Windows path-containment and UTF-8 test-portability issues are fixed, the direct-plan inheritance fallback is now structured, and the complete test/evaluator/compile verification is clean. The remaining qualification is that this is an API/CLI project with no frontend and live Anthropic behavior was not verified.

The score is based on executed behavior and tests, not README claims.

## 2. Assessment scorecard

| Assessment area | Maximum points | Verified status | Evidence | Missing work |
|---|---:|---|---|---|
| Data profiling and cleaning | 18 | PARTIAL | Evaluator: 15/15 cases passed; orders cleaned 1515→1500 with profiles, issue plan, validation, idempotency, hashes and artifacts. Ambiguous dates were nulled rather than guessed. | No general statistical outlier detection/remediation; canonical column names are required and AI-suggested cleaning steps are not implemented. |
| Retail intelligence | 14 | PARTIAL | Executed sales, return-rate, region/category, inventory and safe-join flows on sample data. Metrics are computed from typed frames and include formulas/exclusions. | No dedicated customer/product segmentation workflow; no recommendation/scoring capability; inventory is a snapshot, not a time series. |
| AI chat assistant | 20 | PARTIAL | Live API test completed three-turn West-region/category return-rate flow. Answers contained versions, columns, filters, joins, operations, row counts and bounded results. Clarification, refusal, invalid-plan and injection-question paths were exercised. | Real external-model behavior was not verified; mock/rule phrasing support is intentionally limited. |
| Safe execution and AI engineering | 12 | PARTIAL | Pydantic plans, explicit allow-lists, declared joins, bounded limits, no dynamic SQL/code, prompt catalog sanitization, retries, breaker and deterministic fallback are implemented and tested. | Query timeout is explicitly soft because worker threads cannot be killed; resource control is therefore not a hard execution boundary. |
| Batch evaluator and reliability | 10 | PARTIAL | `python -m app.evaluate ... --strict`: 15 passed, 174/174 checks, 22 questions, 30/33 datasets cleaned, exit code 0. Outputs and artifacts were generated in an isolated temporary directory; concurrent submission passed 10/10 runs. | No live-model benchmark; the evaluator remains single-process and bounded by whole-file limits. |
| Architecture and code quality | 10 | PARTIAL | Clear pipeline/storage/retail/chat/LLM/evaluator separation; API is a thin service layer; compileall passed. | No frontend exists; no UI/browser flow can be audited. Logging is mostly event persistence/console demos rather than structured application logging. |
| Security and privacy | 6 | PARTIAL | No committed `.env`; `.gitignore` excludes runtime data; owner scoping, filename/size checks, hash verification, path checks, PII catalog restrictions and plan validation are present. Cross-owner API access returned 404. | Authentication defaults to caller-supplied `X-Owner-Id` unless `API_TOKEN` is configured; this is documented as proxy-dependent, not sufficient standalone production authentication. |
| Automated tests | 5 | PASS | Normal `python -m pytest -q` passed 373 tests; focused regression tests passed 9/9; concurrent identical-submission test passed 10/10. | Coverage still cannot prove live-provider quality. |
| README and technical reasoning | 5 | PASS | README now includes Windows PowerShell setup/run/evaluator commands, API/CLI-only scope, mock-provider default, live-provider verification limitation and authentication caveat. | No frontend documentation is needed because the scope is explicitly API/CLI. |
| **TOTAL** | **100** |  |  | **88** |

## 3. Repository and architecture findings

### Implemented

- Python FastAPI backend in `app/api/main.py`.
- Deterministic pipeline: loader, profiler, issue detector, cleaning planner, cleaner and independent validator.
- SQLite metadata/state plus filesystem artifacts.
- Retail metrics and inventory calculations with declared positional joins.
- Chat plan schema, validator, executor, grounding, fallback rule planner and persisted sessions.
- Batch evaluator using the same production services as the API.
- Sample datasets, edge cases and 373 tests (371 original tests plus two regression tests added during this fix).

### Not implemented or explicitly limited

- No frontend files were found (`.html`, `.js`, `.jsx`, `.ts`, `.tsx`, `.css` search returned no matches). The project exposes an HTTP API and scripts only.
- AI-suggested cleaning steps are not applied; cleaning is code-planned and deterministic.
- No column-alias mapping for foreign schemas; canonical names such as `orders.quantity` are required.
- No recommendation/performance scoring capability.
- No retention/cleanup policy for old runs.
- Anthropic adapter is tested against a fake client, not a live API.

## 4. Commands executed and results

All commands were run from:

`C:\Users\amirh\Downloads\retail-workbench\retail-workbench`

### Dependency/configuration

1. `python -m pip install python-multipart`
   - Installed a dependency already declared in `requirements.txt`.
   - This was necessary because the existing environment did not contain it and FastAPI could not construct the multipart upload route.

### Automated tests

1. `python -m pytest -q`
   - **373 passed in 68.81s** with no `PYTHONUTF8` workaround.
2. Focused regression command covering docs, path safety, concurrency, inheritance and multi-turn context:
   - **9 passed**.
3. `python -m pytest -q tests/test_storage.py::test_concurrent_identical_submits_produce_exactly_one_run` repeated 10 times:
   - **Run 1 PASS, Run 2 PASS, Run 3 PASS, Run 4 PASS, Run 5 PASS, Run 6 PASS, Run 7 PASS, Run 8 PASS, Run 9 PASS, Run 10 PASS**.
   - Summary: **10 passed, 0 failed**.
4. `python -m compileall -q app scripts tests`
   - `COMPILE_EXIT=0`.

### Batch evaluator

```text
python -m app.evaluate --input cases.json --output <temp>\results.json --artifacts-dir <temp>\artifacts --strict
```

Result:

```text
cases: 15 passed, 0 failed, 0 error / 15
datasets: 30/33 cleaned
questions: 22
checks: 174/174
exit code: 0
```

The evaluator produced cleaned CSVs, reports and chat traces without touching the repository datasets.

### Walkthrough

`python scripts/walkthrough.py 1`

Passed. It cleaned `orders.csv` from 1515 to 1500 rows, reported issues and decisions, validated row loss, and confirmed idempotency.

### API/runtime verification

1. Started Uvicorn:

   `python -m uvicorn app.api.main:create_app --factory --host 127.0.0.1 --port 8765`

2. `/health` returned HTTP 200 and reported `llm_provider: mock`.
3. In-process FastAPI client flow uploaded `orders.csv`, `products.csv`, `stores.csv` and `customers.csv`.
4. Run completed successfully:
   - customers: 204 → 200
   - orders: 1515 → 1500
   - products: 63 → 60
   - stores: 8 → 8
5. Report and cleaned-download endpoints returned HTTP 200.
6. Created a chat session and executed:
   - “Show total sales by region”
   - “Only consider the West region”
   - “Among those stores, which categories have the highest return rate?”
7. The second turn retained `region = West`; the third retained that filter and regrouped by category.
8. Prompt-injection question produced a clarification, not a system-prompt disclosure.
9. A malicious aggregate plan attempting `sum(order_id)` returned `INVALID_AGGREGATION`.
10. The same run requested under another owner returned HTTP 404.

No browser verification was performed because the repository contains no frontend application to open.

## 5. Requirement-by-requirement findings

### A. Data profiling and cleaning

Verified:

- CSV ingestion with UTF-8/cp1252/Latin-1 fallback.
- Size/type/name validation and original-byte preservation.
- Before/after profiles with nulls, placeholders, duplicates, inferred types, parse failures, variants, numeric/date statistics and PII redaction.
- Stable issue and plan IDs.
- Cleaning execution logs with changed/nulled cells, removed rows and examples.
- Ambiguous date handling that avoids guessing and records nulled values.
- Raw and cleaned artifacts, SHA-256 hashes, data versions and read-back comparison.
- Independent validation and idempotency checks.
- Downloadable cleaned artifacts through the API and evaluator.

Gaps:

- Negative quantities are detected/flagged rather than corrected, which is an acceptable safety choice but not full outlier remediation.
- Data is read whole in memory up to the configured cap.
- Column schemas must use the project’s canonical names.

### B. Retail intelligence

Verified from actual sample data:

- Sales/revenue/order/AOV/margin/cancellation/return metrics.
- Region/store/category/brand/product/month dimensions where required datasets are loaded.
- Return-rate investigation after a West-region follow-up.
- Inventory stock-out, low-stock, sell-through and ageing metrics.
- Store/product/customer declared joins with fan-out refusal and orphan reporting.

Gaps:

- No dedicated customer segmentation or recommendation scoring.
- Inventory stock-out is only a point-in-time share because the input has no timestamp history.

### C. Chat assistant

Verified:

- Filtering, sorting, grouping, aggregation, distinct and descriptive statistics.
- Top/bottom bounded results.
- Multi-turn filter inheritance merged in code.
- Declared, validated joins.
- Evidence including dataset versions, columns, filters, joins, operations, row counts, previews, formulas and caveats.
- Deterministic numeric answers and grounding checks.
- Clarification and unanswerable plan types.
- Prompt injection in data cells is treated as data and removed from model catalog labels.
- Malicious fields/operators/aggregations/limits and unauthorized joins are rejected.
- Retry, circuit-breaker and deterministic fallback paths are covered by tests/evaluator.
- Explicit cleaned-data session pinning; raw-data requests clarify rather than silently switching versions.
- Persisted question/plan/trace/result-preview/answer session records.

Gaps:

- The real Anthropic service was not exercised because no API key was available/used. The verified runtime provider was the deterministic mock.
- The mock/rule planner intentionally supports a fixed vocabulary and does not provide broad natural-language coverage.

### D. Safe execution and AI engineering

The model emits JSON plans only. Pydantic parsing, semantic allow-list validation and fixed pandas/retail executors prevent arbitrary Python or SQL execution. Limits include filters, joins, group fields, result rows and a scan cap. The main weakness is the documented soft timeout: timed-out executor threads remain alive and cannot be forcibly terminated.

### E. Batch evaluator and reliability

The shipped evaluator is measurable, deterministic and continues after per-case failures. It generated expected artifacts and all 174 shipped checks passed. The fixed Windows path handling was stress-tested successfully in all 10 required concurrent-submission runs.

### F. Architecture and code quality

Responsibilities are well separated and the API is thin. Error payloads avoid stack traces and internal paths. The most material architecture gap is that there is no frontend despite the requested audit scope including frontend startup/browser behavior.

### G. Security and privacy

No committed credentials were found; `.env` is ignored and `.env.example` contains placeholders. Uploads are constrained, paths are checked, artifacts are hash-verified, owners scope all reads, and model catalogs exclude PII/identifier values. The security model is not standalone authentication: without `API_TOKEN`, the caller supplies the owner identity. It is suitable only behind the documented authenticating proxy.

### H. Automated tests

Coverage is broad: cleaning, profiling, joins, metrics, validator rejection, multi-turn state, prompt injection, model failures, grounding, evaluator determinism, isolation, API errors and large inputs are represented. The normal Windows command now passes, and the concurrent idempotent submission test passed in all 10 required stress runs.

### I. README

The README is unusually detailed and documents the AI/code boundary, data sent to providers, security decisions, limitations, evaluator format and example flows. It now includes exact PowerShell commands and explicitly states the API/CLI-only scope, mock default, live-provider limitation and authentication model.

## 6. Security and runtime issues requiring attention

### Medium priority

1. **No frontend**
   - There is no browser UI, frontend build, or frontend API integration to submit or demonstrate.

2. **Authentication is deployment-dependent**
   - `X-Owner-Id` is caller supplied unless `API_TOKEN` is configured.
   - This is documented, but should be made explicit in submission materials as a non-production standalone auth model.

3. **Soft query timeout**
   - A timed-out worker is cancelled only at the future level; the underlying Python thread may continue using memory/CPU.

4. **Live provider behavior remains unverified**
   - The shipped and verified provider is the deterministic mock; Anthropic is not exercised without credentials.

## 7. Exact files/locations changed

- `app/storage/paths.py:safe_join`: fixed with consistent Windows extended-prefix normalization and OS-level commonpath checking.
- `tests/test_docs.py`: fixed documentation reads to use explicit UTF-8.
- `README.md`: added PowerShell setup/run/evaluator commands and clarified API/CLI, provider and authentication scope.
- `app/chat/planner.py` and `app/chat/service.py`: malformed direct inheritance now raises structured `INVALID_PLAN` instead of silently bypassing inheritance.
- `tests/test_storage.py` and `tests/test_chat_service.py`: added path and inheritance regression coverage.

## 8. Priority-wise recommended fixes

### Completed P0/P1 fixes

1. Fixed and stress-tested Windows `safe_join` (10/10 passes).
2. Made documentation reads explicit UTF-8.
3. Removed the silent direct-plan inheritance fallback and added structured validation.
4. Added Windows PowerShell commands and clarified API/CLI/provider/authentication scope.

### P2 — improvement after submission gate

1. Add schema/column alias mapping.
2. Add customer/product segmentation capabilities if required by the rubric.
3. Replace soft timeouts with an enforceable process/resource boundary for untrusted expensive work.
4. Add retention and cleanup for old run artifacts.

## 9. Submission readiness

**Current verdict: READY FOR SUBMISSION WITH DOCUMENTED LIMITATIONS.**

The core implementation is credible and demonstrable, and the deterministic mock path makes the project reviewable without paid services. The required test, stress, evaluator and compile checks now pass. If the assessment explicitly requires a frontend, this API/CLI-only repository remains incomplete in that separate scope; live Anthropic behavior is also intentionally unverified.
