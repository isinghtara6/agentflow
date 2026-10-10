from prometheus_client import Counter, Gauge, Histogram

REQS = Counter("agentflow_requests_total", "HTTP requests", ["endpoint", "status"])
LLM_CALLS = Counter("agentflow_llm_calls_total", "LLM calls", ["agent", "model", "outcome"])
LLM_TOKENS = Counter("agentflow_llm_tokens_total", "LLM tokens", ["model", "direction"])
LLM_LAT = Histogram("agentflow_llm_latency_seconds", "LLM latency", ["agent"],
                    buckets=(0.5, 1, 2, 5, 10, 20, 40, 60))
CACHE = Counter("agentflow_cache_total", "Cache lookups", ["result"])
GUARD = Counter("agentflow_guardrail_blocks_total", "Guardrail blocks", ["stage", "reason"])
RATE_LIMITED = Counter("agentflow_rate_limited_total", "Rate limited", ["scope"])
RUN_LAT = Histogram("agentflow_run_latency_seconds", "Workflow latency",
                    buckets=(1, 2, 5, 10, 20, 40, 80, 150))
INFLIGHT = Gauge("agentflow_inflight_runs", "In-flight workflow runs")
