"""HTTP API over the same services the evaluator uses. Run:  uvicorn app.api.main:create_app --factory --port 8000

Identity: the caller supplies `X-Owner-Id` (isolation key). Put this behind an authenticating proxy, or set API_TOKEN to
require `Authorization: Bearer <token>`. Run ONE process (in-process thread pool, caches and circuit breaker are per process).
Errors are always {"error": {code, message, details}}; unexpected failures never leak internals."""
import hmac
import json
import os
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict

from app.chat.service import ChatService
from app.config import settings
from app.errors import PipelineError
from app.llm.base import get_provider
from app.storage.paths import validate_owner
from app.storage.runs import MAX_FILES, PIPELINE_VERSION, RunService

HTTP = {"RUN_NOT_FOUND": 404, "SESSION_NOT_FOUND": 404, "DATASET_NOT_FOUND": 404, "REPORT_NOT_AVAILABLE": 404, "DATASET_NOT_READY": 409,
        "RUN_NOT_READY": 409, "FILE_TOO_LARGE": 413, "REQUEST_TOO_LARGE": 413, "AUTH_REQUIRED": 401, "ARTIFACT_CORRUPT": 500}
_INTERNAL_KEYS = ("original_path", "cleaned_path", "report_path")


class AskBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Optional[str] = None
    plan: Optional[dict] = None


def _public_run(run: dict) -> dict:
    run = dict(run)
    run["datasets"] = [{k: v for k, v in d.items() if k not in _INTERNAL_KEYS} for d in run["datasets"]]
    return run


def create_app(root=None, provider=None, max_upload_mb: float | None = None, api_token: str | None = None, recover: bool = True) -> FastAPI:
    runs = RunService(root=root, max_upload_mb=max_upload_mb)
    if recover:
        runs.recover_stale_runs()                      # single-process assumption: anything 'running' at startup was interrupted
    prov = provider if provider is not None else get_provider()
    chat = ChatService(runs, prov)
    token = api_token if api_token is not None else os.getenv("API_TOKEN", "")
    app = FastAPI(title="Retail Data & Conversational Intelligence Workbench", version=PIPELINE_VERSION)
    app.state.runs, app.state.chat = runs, chat

    @app.exception_handler(PipelineError)
    async def pipeline_error(_: Request, e: PipelineError):
        return JSONResponse({"error": e.to_dict()}, status_code=HTTP.get(e.code, 400))

    @app.exception_handler(Exception)
    async def unexpected(_: Request, e: Exception):
        return JSONResponse({"error": {"code": "INTERNAL_ERROR", "message": "Unexpected server error.", "details": {}}}, status_code=500)

    def owner(x_owner_id: Optional[str] = Header(default=None), authorization: Optional[str] = Header(default=None)) -> str:
        if token and not (authorization and hmac.compare_digest(authorization, f"Bearer {token}")):
            raise PipelineError("AUTH_REQUIRED", "A valid bearer token is required.")
        return validate_owner(x_owner_id or "")

    @app.get("/health")
    def health():
        return {"status": "ok", "pipeline_version": PIPELINE_VERSION, "llm_provider": prov.name}

    @app.post("/runs", status_code=202)
    async def create_run(request: Request, files: list[UploadFile] = File(...), decisions: Optional[str] = Form(default=None), who: str = Depends(owner)):
        limit = int(runs.max_upload_mb * 1024 * 1024)
        if len(files) > MAX_FILES:
            raise PipelineError("TOO_MANY_FILES", f"At most {MAX_FILES} files per run.")
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_FILES * limit + 1_000_000:
            raise PipelineError("REQUEST_TOO_LARGE", "Request body is too large.")
        try:
            dec = json.loads(decisions) if decisions else {}
        except ValueError:
            raise PipelineError("INVALID_DECISION", "'decisions' must be a JSON object {step_id: 'apply'|'skip'}.")
        if not isinstance(dec, dict):
            raise PipelineError("INVALID_DECISION", "'decisions' must be a JSON object.")
        uploads = []
        for f in files:
            data = await f.read(limit + 1)              # never buffer more than limit+1 bytes per file; the service rejects the oversize one
            uploads.append((f.filename or "upload.csv", data))
        r = await run_in_threadpool(runs.submit, who, uploads, dec, False)
        return JSONResponse(r, status_code=202, headers={"Location": f"/runs/{r['run_id']}"})

    @app.get("/runs")
    def list_runs(who: str = Depends(owner)):
        return {"runs": runs.list_runs(who)}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str, who: str = Depends(owner)):
        return _public_run(runs.get_run(who, run_id))

    @app.get("/runs/{run_id}/datasets/{name}/report")
    def get_report(run_id: str, name: str, who: str = Depends(owner)):
        rep = runs.get_report(who, run_id, name)
        rep.pop("run", None)
        rep["artifacts"] = {k: v for k, v in rep["artifacts"].items() if k in ("original_sha256", "cleaned_sha256")}
        return rep

    @app.get("/runs/{run_id}/datasets/{name}/cleaned")
    def download_cleaned(run_id: str, name: str, who: str = Depends(owner)):
        return Response(runs.cleaned_bytes(who, run_id, name), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}.cleaned.csv"'})

    @app.post("/runs/{run_id}/sessions", status_code=201)
    def create_session(run_id: str, who: str = Depends(owner)):
        return chat.create_session(who, run_id)

    @app.get("/sessions")
    def list_sessions(who: str = Depends(owner)):
        return {"sessions": chat.list_sessions(who)}

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str, who: str = Depends(owner)):
        return chat.get_session(who, session_id)

    @app.post("/sessions/{session_id}/ask")
    def ask(session_id: str, body: AskBody, who: str = Depends(owner)):
        return chat.ask(who, session_id, question=body.question, plan=body.plan)

    return app
