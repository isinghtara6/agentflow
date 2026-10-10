import asyncio
import hashlib
import hmac
import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis

from .cache import Cache
from .config import Settings, get_settings
from .guardrails import GuardrailBlocked, Guardrails
from .llm import LLM, BudgetExceeded, CircuitOpen
from .logging_setup import request_id_var, setup_logging
from .metrics import INFLIGHT, RATE_LIMITED, REQS
from .rate_limiter import DailyQuota, RateLimited, RateLimiter
from .schemas import RunRequest, RunResponse
from .workflow import Workflow

log = logging.getLogger("agentflow")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    setup_logging(s.log_level)
    redis = Redis.from_url(s.redis_url, decode_responses=True, socket_timeout=3,
                           socket_connect_timeout=3, health_check_interval=30)
    limiter = RateLimiter(redis, s.rate_limit_fail_open)
    llm = LLM(s, limiter)
    guards = Guardrails(s, llm)
    app.state.s, app.state.redis, app.state.limiter = s, redis, limiter
    app.state.quota = DailyQuota(redis, s.daily_token_quota)
    app.state.workflow = Workflow(s, llm, Cache(redis, s.cache_ttl_s), guards)
    app.state.inflight = asyncio.Semaphore(s.max_inflight_runs)
    yield
    await redis.aclose()


app = FastAPI(title="AgentFlow", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request_id_var.set(rid)
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


# ---------- error mapping ----------
def _err(status: int, msg: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": msg, "request_id": request_id_var.get()}, status_code=status, headers=headers)


@app.exception_handler(GuardrailBlocked)
async def _guard(_, e: GuardrailBlocked):
    return _err(400 if e.stage == "input" else 422, e.public_msg)


@app.exception_handler(RateLimited)
async def _rl(_, e: RateLimited):
    return _err(429, "Rate limit exceeded", {"Retry-After": str(max(1, int(e.retry_after_s + 0.99)))})


@app.exception_handler(CircuitOpen)
async def _cb(_, e):
    return _err(503, "Upstream model temporarily unavailable", {"Retry-After": "30"})


@app.exception_handler(BudgetExceeded)
async def _bud(_, e):
    return _err(422, "Request too complex for the token budget")


@app.exception_handler(asyncio.TimeoutError)
async def _to(_, e):
    return _err(504, "Workflow timed out")


# ---------- auth + admission control ----------
async def auth(x_api_key: str | None = Header(default=None), s: Settings = Depends(get_settings)) -> str:
    keys = s.api_key_list
    if not keys:
        if s.env == "prod":
            raise HTTPException(500, "server misconfigured: no API keys")
        return "anonymous"
    if not x_api_key or not any(hmac.compare_digest(x_api_key, k) for k in keys):
        raise HTTPException(401, "invalid API key")
    return hashlib.sha256(x_api_key.encode()).hexdigest()[:12]


async def admit(request: Request, user: str) -> dict:
    st = request.app.state
    r = await st.limiter.check(f"rl:user:{user}", st.s.user_rpm, st.s.user_burst)
    headers = {"X-RateLimit-Limit": str(r.limit), "X-RateLimit-Remaining": str(int(r.remaining))}
    if not r.allowed:
        RATE_LIMITED.labels("user").inc()
        raise RateLimited(r.retry_after_s, "user")
    if await st.quota.remaining(user) <= 0:
        RATE_LIMITED.labels("quota").inc()
        raise HTTPException(429, "daily token quota exhausted")
    if st.inflight.locked():
        raise HTTPException(503, "server busy", headers={"Retry-After": "5"})
    return headers


# ---------- endpoints ----------
@app.post("/v1/run", response_model=RunResponse)
async def run(body: RunRequest, request: Request, response: Response, user: str = Depends(auth)):
    st = request.app.state
    response.headers.update(await admit(request, user))
    async with st.inflight:
        INFLIGHT.inc()
        try:
            res = await st.workflow.run(body.query, body.style)
        finally:
            INFLIGHT.dec()
    await st.quota.add(user, res.tokens_used)
    REQS.labels("run", "200").inc()
    log.info("run done", extra={"extra_fields": {"user": user, "tokens": res.tokens_used,
                                                 "cached": res.cached, "latency_ms": res.latency_ms}})
    return RunResponse(request_id=request_id_var.get(), **res.__dict__)


@app.post("/v1/run/stream")
async def run_stream(body: RunRequest, request: Request, user: str = Depends(auth)):
    """Server-Sent Events: progress events, then a final `result` (or `error`) event."""
    st = request.app.state
    headers = await admit(request, user)
    q: asyncio.Queue = asyncio.Queue()

    async def emit(event: str, data: dict):
        await q.put((event, data))

    async def worker():
        try:
            async with st.inflight:
                res = await st.workflow.run(body.query, body.style, emit)
            await st.quota.add(user, res.tokens_used)
            await q.put(("result", {"request_id": request_id_var.get(), **res.__dict__}))
        except GuardrailBlocked as e:
            await q.put(("error", {"status": 400, "error": e.public_msg}))
        except (RateLimited, CircuitOpen):
            await q.put(("error", {"status": 503, "error": "temporarily unavailable"}))
        except TimeoutError:
            await q.put(("error", {"status": 504, "error": "timed out"}))
        except Exception:
            log.exception("stream run failed")
            await q.put(("error", {"status": 500, "error": "internal error"}))
        finally:
            await q.put(None)

    task = asyncio.create_task(worker())

    async def gen():
        try:
            while (item := await q.get()) is not None:
                yield f"event: {item[0]}\ndata: {json.dumps(item[1])}\n\n"
        finally:
            if not task.done():
                task.cancel()  # client disconnected -> stop spending tokens

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={**headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request):
    try:
        await request.app.state.redis.ping()
    except Exception:
        return JSONResponse({"status": "redis_unavailable"}, status_code=503)
    return {"status": "ready"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
