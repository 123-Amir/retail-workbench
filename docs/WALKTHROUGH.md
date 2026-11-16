# 3–5 minute walkthrough

Run this once before the demo:

```bash
python scripts/walkthrough.py
```

The script uses the sample data and the same services as the evaluator.

| Time | Step | Command | Show |
|---|---|---|---|
| 0:00–0:25 | Overview | — | Explain the boundary: the model proposes a JSON plan; Python validates, executes and checks it. Show README §2. |
| 0:25–1:25 | Clean one file | `python scripts/walkthrough.py 1` | `orders.csv` goes from 1515 to 1500 rows. Show issues, stable step IDs, validation, and idempotency. |
| 1:25–2:10 | Handle ambiguity | `python scripts/walkthrough.py 2` | Show that ambiguous dates are not guessed. Dates without enough evidence become null and are reported. |
| 2:10–3:00 | Retail metrics | `python scripts/walkthrough.py 3` | Show safe joins, orphan keys, fan-out refusal, category revenue, formulas, exclusions, and the inventory snapshot caveat. |
| 3:00–4:00 | Multi-turn chat | `python scripts/walkthrough.py 4` | Ask the three-turn West-region example. Show preserved filters, evidence, clarification, refusal, and `chat_trace.json`. |
| 4:00–4:45 | Failure handling | `python scripts/walkthrough.py 5` | Show model outage fallback, rejected invented numbers, rejected hostile plans, and validator-protected artifacts. |
| 4:45–5:00 | Evaluation | `python -m app.evaluate --input cases.json --output results.json --artifacts-dir ./evaluation_output` | Explain that the evaluator uses the same pipeline and produces reports, cleaned files, and chat traces. |

Open these files before presenting:

- `evaluation_output/clean-orders/orders.cleaned.csv`
- `evaluation_output/clean-orders/orders.report.json`
- `evaluation_output/retail-multi-dataset-conversation/chat_trace.json`

Optional live-provider demo:

```bash
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=... python scripts/chat_demo.py
```

Live Anthropic behavior is not verified by the repository tests.
