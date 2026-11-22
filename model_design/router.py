from typing import Literal
import dspy

class QueryIntentSignature(dspy.Signature):
    """
    Determine whether a query requires web search or can be
    answered reliably using internal model knowledge.

    Choose 'search' for:
    - Current information, recent events, or live data.
    - Information likely to have changed, such as laws,
      leadership, prices, schedules, or software versions.
    - Exact URLs, current documentation, or multi-source research.

    Choose 'static' for:
    - Established general, historical, and scientific knowledge.
    - Self-contained math, logic, and coding tasks.
    - Creative writing, translation, and brainstorming.
    - Analyzing, summarizing, or rewriting provided content.

    Classify based on whether the answer requires up-to-date
    external information, not merely the subject or wording
    of the question.

    Do not search when the answer can be reliably established
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