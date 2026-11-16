# Review notes

## Architecture in one minute

The project has two main parts:

- **Data:** load CSVs as strings, profile them, plan and apply deterministic cleaning, validate the result, and publish hashed artifacts.
- **Chat:** turn a question into a JSON plan, validate it against the loaded data, execute fixed operations, and return evidence.

`RunService` and `ChatService` are the main service entry points. The API and evaluator call those services.

## AI boundary

- The model chooses a plan. It does not see raw rows.
- Python performs joins, filters, aggregations, validation, and metric arithmetic.
- The default answer is a template over computed results.
- Optional model explanations must pass numeric grounding.
- Invalid, ambiguous, or unsupported questions are rejected or clarified.
- A model outage falls back to the deterministic rule planner.

## Design choices to explain

1. **Fixed-point cleaning:** rows that become identical after lossy cleaning are merged and reported, so a second cleaning run changes nothing.
2. **Positional joins:** dimension keys must be unique. This prevents fan-out and keeps row order stable.
3. **Code-owned follow-ups:** the model marks a follow-up; Python carries forward and replaces filters.
4. **Strings first:** the profiler decides types instead of letting pandas silently change values such as `007`.
5. **Independent validation:** validation compares real before/after frames, not just executor logs.
6. **Simple storage:** SQLite, filesystem artifacts, and an in-process thread pool keep setup small but are not horizontally scalable.

## Common questions

- **Why not allow model-generated SQL or Python?** Plans are easier to validate and replay, and cannot execute arbitrary code.
- **How are invented numbers blocked?** Numeric grounding discards explanations containing numbers absent from the result or evidence.
- **What happens to prompt injection in a cell?** It remains data. Suspicious labels are removed from model prompts.
- **How are users isolated?** Every run and session is scoped by owner. Missing and unauthorized resources return the same error.
- **What breaks first at scale?** Whole-file reads, in-process jobs, per-process caches, and SQLite. A queue and larger database would be next steps.

## Known limitations

- No frontend; this is an API/CLI project.
- Files are read into memory up to `MAX_UPLOAD_MB`.
- Query timeouts are soft because Python worker threads cannot be killed.
- Retail inputs need canonical dataset and column names.
- Inventory is a snapshot, not a time series.
- The mock provider supports a fixed set of phrases.
- Live Anthropic behavior is not verified.
- AI-suggested cleaning steps and recommendation scoring are not implemented.
