from typing import Literal
import dspy

class QueryIntentSignature(dspy.Signature):
    """
    Determine whether a query requires web search or can be
    answered using internal model knowledge.

    Choose 'search' if the query requires:
    - Current information, recent events, or live data.
    - Multi-source research or cross-referencing.
    - Exact URLs, current specifications, schedules, or fact-checking.
    - Information likely to have changed, such as laws, leadership,
      software documentation, or API behavior.

    Choose 'static' for:
    - Established general, historical, or scientific knowledge.
    - Self-contained math, logic, and coding tasks.
    - Creative writing, translation, and brainstorming.
    - Summarizing, rewriting, or analyzing user-provided content.

    Search requirements override task type.
    Do not search when the query can be reliably answered
    using internal knowledge or provided context.
    """
    question: str = dspy.InputField(
        desc="The user query or task to route."
    )
    intent: Literal["search", "static"] = dspy.OutputField(
        desc="Must be either search or static."
    )

class RouteSAGRouter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(QueryIntentSignature)

    def forward(self, question: str):
        result = self.classify(question=question)
        return result.intent.strip().lower()