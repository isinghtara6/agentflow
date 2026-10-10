"""Orchestrator: guardrails -> cache(single-flight) -> plan -> parallel research -> write -> review loop -> output guard."""
import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace

from .agents import PLANNER, RESEARCHER, REVIEWER, WRITER, Plan, Review, parse_json
from .cache import Cache
from .config import Settings
from .guardrails import Guardrails
from .llm import LLM, BudgetExceeded, CircuitOpen, RunBudget
from .metrics import RUN_LAT

log = logging.getLogger(__name__)
Emit = Callable[[str, dict], Awaitable[None]]


async def _noop(event: str, data: dict) -> None:
    return None


@dataclass
class RunResult:
    answer: str
    subtasks: list[str]
    review_rounds: int
    tokens_used: int
    cached: bool
    redactions: list[str] = field(default_factory=list)
    latency_ms: int = 0


class Workflow:
    def __init__(self, settings: Settings, llm: LLM, cache: Cache, guards: Guardrails):
        self.s, self.llm, self.cache, self.guards = settings, llm, cache, guards

    async def run(self, query: str, style: str = "concise", emit: Emit = _noop) -> RunResult:
        t0 = time.monotonic()
        await emit("guardrails", {"stage": "input"})
        checked = await self.guards.check_input(query)

        key = self.cache.make_key("run", {
            "v": self.s.prompt_version, "q": " ".join(checked.text.split()), "style": style,
            "m": [self.s.strong_model, self.s.fast_model],
        })

        async def compute() -> dict:
            return await asyncio.wait_for(self._execute(checked.text, style, emit), self.s.workflow_timeout_s)

        data, hit = await self.cache.get_or_compute(key, compute)
        if hit:
            await emit("cache_hit", {})
        elapsed = time.monotonic() - t0
        RUN_LAT.observe(elapsed)
        return RunResult(
            answer=data["answer"], subtasks=data["subtasks"], review_rounds=data["review_rounds"],
            tokens_used=0 if hit else data["tokens_used"], cached=hit,
            redactions=checked.redactions, latency_ms=int(elapsed * 1000),
        )

    async def _execute(self, query: str, style: str, emit: Emit) -> dict:
        s, g = self.s, self.guards
        budget = RunBudget(s.run_token_budget)
        user_block = f"<user_request>\n{query}\n</user_request>\nStyle: {style}"

        # 1. plan
        await emit("planning", {})
        planner = replace(PLANNER, role=PLANNER.role.format(n=s.max_subtasks))
        plan = self._parse_plan(await planner.run(self.llm, s, g.canary, user_block, budget))
        await emit("planned", {"subtasks": [t.title for t in plan.subtasks]})

        # 2. parallel research with bounded concurrency + failure isolation
        sem = asyncio.Semaphore(s.agent_concurrency)

        async def research(t):
            async with sem:
                await emit("researching", {"id": t.id, "title": t.title})
                return await RESEARCHER.run(
                    self.llm, s, g.canary,
                    f"{user_block}\n\nSubtask: {t.title}\n{t.instructions}", budget,
                )

        results = await asyncio.gather(*(research(t) for t in plan.subtasks), return_exceptions=True)
        for r in results:  # infrastructure failures must abort, not silently degrade
            if isinstance(r, (BudgetExceeded, CircuitOpen, asyncio.CancelledError)):
                raise r
        notes = []
        for t, r in zip(plan.subtasks, results, strict=True):
            body = r if isinstance(r, str) else "(research unavailable for this subtask)"
            notes.append(f"## {t.title}\n{body}")
        if all(isinstance(r, Exception) for r in results):
            raise RuntimeError("all research subtasks failed")
        notes_block = "<research_notes>\n" + "\n\n".join(notes) + "\n</research_notes>"

        # 3. write
        await emit("writing", {})
        draft = await WRITER.run(self.llm, s, g.canary, f"{user_block}\n\n{notes_block}", budget)

        # 4. review loop (reviewer failure -> accept draft; output guard is the final safety net)
        rounds = 0
        for _ in range(s.max_review_rounds):
            await emit("reviewing", {})
            try:
                review = parse_json(
                    await REVIEWER.run(self.llm, s, g.canary,
                                       f"{user_block}\n\n{notes_block}\n\n<draft>\n{draft}\n</draft>", budget),
                    Review,
                )
            except (ValueError, BudgetExceeded):
                break
            rounds += 1
            if review.approved:
                break
            issues = "\n".join(f"- {i}" for i in review.issues[:8])
            draft = await WRITER.run(
                self.llm, s, g.canary,
                f"{user_block}\n\n{notes_block}\n\n<draft>\n{draft}\n</draft>\n<issues>\n{issues}\n</issues>\n"
                "Rewrite the draft fixing every issue.", budget,
            )

        # 5. output guardrail
        answer = g.check_output(draft)
        return {"answer": answer, "subtasks": [t.title for t in plan.subtasks],
                "review_rounds": rounds, "tokens_used": budget.used}

    def _parse_plan(self, raw: str) -> Plan:
        plan = parse_json(raw, Plan)
        plan.subtasks = plan.subtasks[: self.s.max_subtasks]
        return plan
