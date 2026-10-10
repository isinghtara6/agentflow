from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "dev"
    log_level: str = "INFO"
    anthropic_api_key: str = ""
    api_keys: str = ""  # comma-separated
    redis_url: str = "redis://localhost:6379/0"

    # models: cheap/fast for planning, review and policy checks; strong for research + writing
    strong_model: str = "claude-sonnet-5-5"
    fast_model: str = "claude-haiku-5-5"

    # rate limiting (token bucket)
    user_rpm: int = 20
    user_burst: int = 5
    global_llm_rpm: int = 600          # protects your Anthropic org limits across ALL pods
    global_llm_burst: int = 50
    daily_token_quota: int = 500_000   # per user per UTC day
    rate_limit_fail_open: bool = True  # if Redis is down, serve instead of blocking

    # cache
    cache_ttl_s: int = 3600
    prompt_version: str = "v1"         # bump to invalidate cache when prompts change

    # guardrails
    max_input_chars: int = 6000
    llm_policy_check: bool = True
    block_on_injection: bool = True

    # workflow
    max_subtasks: int = 4
    max_review_rounds: int = 1
    agent_concurrency: int = 4
    max_inflight_runs: int = 20        # per pod; excess -> 503 (load shedding)
    run_token_budget: int = 60_000     # hard cap per run
    workflow_timeout_s: float = 150
    llm_timeout_s: float = 60
    llm_max_retries: int = 4
    breaker_threshold: int = 5
    breaker_cooldown_s: float = 30

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
