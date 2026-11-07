import asyncio
import time
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.solver import solver, TaskState, Generate, Solver
from inspect_ai.scorer import Score, Scorer, scorer, accuracy, mean, model_graded_qa

from inspect_evals.simpleqa import simpleqa_verified
from inspect_evals.browse_comp import browse_comp

from router import RouteSAGRouter
from model import init_dspy_model
from search import ddg_search

init_dspy_model()
router = RouteSAGRouter()

def combined_routesag_dataset():
    samples = []
    sources = [(simpleqa_verified(), "SimpleQA", "local"), (browse_comp(), "BrowseComp", "search")]
    for task_obj, name, intent in sources:
        for idx, sample in enumerate(task_obj.dataset):
            meta = {**(sample.metadata or {}), "source_dataset": name, "expected_intent": intent}
            samples.append(sample.model_copy(update={"id": f"{name.lower()}_{sample.id or idx}", "metadata": meta}))
    return MemoryDataset(samples=samples, name="RouteSAG_Combined")

@solver
def routesag_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        start = time.perf_counter()
        q = state.input_text
        
        raw = str(router(q)).strip().lower()
        intent = raw if raw in ("local", "search") else "local"
        state.metadata["predicted_intent"] = raw if intent == raw else "invalid"

        if intent == "search":
            state.metadata["search_triggered"] = True
            try:
                ctx = await asyncio.to_thread(ddg_search, q)
                ctx = ctx if ctx else "No search results found."
            except Exception as e:
                ctx = f"Search unavailable: {e}"
            prompt = f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer based on context:"
        else:
            state.metadata["search_triggered"] = False
            prompt = q

        state.user_prompt.text = prompt
        state = await generate(state)
        state.metadata["total_latency_seconds"] = time.perf_counter() - start
        return state
    return solve

@scorer(metrics=[accuracy()])
def routing_intent_scorer() -> Scorer:
    async def score(state: TaskState, target) -> Score:
        pred, exp = state.metadata.get("predicted_intent"), state.metadata.get("expected_intent")
        correct = (pred == exp and exp in ("local", "search"))
        return Score(value=1.0 if correct else 0.0, answer=pred, explanation=f"Pred: {pred} | Exp: {exp}")
    return score

@scorer(metrics=[mean()])
def latency_scorer() -> Scorer:
    async def score(state: TaskState, target) -> Score:
        lat = state.metadata.get("total_latency_seconds", 0.0)
        return Score(value=float(lat), answer=f"{lat:.3f}s", explanation="Total response latency")
    return score

@task
def routesag_combined() -> Task:
    return Task(
        dataset=combined_routesag_dataset(),
        solver=routesag_solver(),
        scorer=[model_graded_qa(), routing_intent_scorer(), latency_scorer()]
    )