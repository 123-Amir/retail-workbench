#!/usr/bin/env bash
# SUGGESTION ONLY. This repository was delivered as a finished tree, so it has no real incremental history.
# Real history should reflect your real work; if you want to publish it in logical layers, this is a sensible order.
# Review each `git add` before running. Usage:  bash docs/commit_plan.sh
set -euo pipefail
git init -q 2>/dev/null || true
c() { msg="$1"; shift; git add "$@"; git commit -q -m "$msg" && echo "committed: $msg"; }

c "chore: project skeleton, config, error taxonomy, env example"       .gitignore .env.example requirements.txt pytest.ini app/__init__.py app/config.py app/errors.py
c "feat(data): seeded messy retail datasets and edge-case files"       scripts/generate_sample_data.py data/
c "feat(pipeline): safe CSV loader, parsers and profiler"              app/pipeline/__init__.py app/pipeline/loader.py app/pipeline/parsers.py app/pipeline/profiler.py tests/__init__.py tests/test_profiler.py scripts/profile_demo.py
c "feat(pipeline): issue detection, variant clustering, cleaning plan" app/pipeline/issues.py app/pipeline/variants.py app/pipeline/planner.py app/pipeline/analyze.py app/retail/__init__.py app/retail/aliases.py tests/test_issues_plan.py scripts/plan_demo.py
c "feat(pipeline): deterministic cleaner, independent validation, idempotency" app/pipeline/cleaner.py app/pipeline/validate.py tests/test_cleaner.py scripts/clean_demo.py
c "feat(storage): sqlite runs, atomic hashed artifacts, idempotent submit, isolation" app/storage/ tests/test_storage.py scripts/run_demo.py
c "feat(retail): declared joins with fan-out guard, filters, sales and inventory metrics" app/retail/ tests/test_retail.py scripts/retail_demo.py
c "feat(chat): plan schema, allow-list validator, executor, evidence" app/chat/__init__.py app/chat/plan.py app/chat/catalog.py app/chat/validator.py app/chat/executor.py tests/test_chat_core.py scripts/chat_core_demo.py
c "feat(llm): providers, retry/backoff, circuit breaker, planner, grounding, sessions" app/llm/ app/chat/rules.py app/chat/planner.py app/chat/grounding.py app/chat/service.py tests/test_llm_planner.py tests/test_chat_service.py scripts/chat_demo.py
c "feat(eval): batch evaluator with expectation engine and sample cases" app/evaluate/ cases/ cases.json evaluation_output/.gitkeep tests/test_evaluate.py
c "feat(api): FastAPI layer over the same services"                    app/api/ tests/test_api.py
c "docs: README, walkthrough, review notes, requirements traceability" README.md docs/ scripts/walkthrough.py tests/test_docs.py
