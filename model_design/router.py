from typing import Literal
import dspy

class QueryIntentSignature(dspy.Signature):
  """Determine if answering the question requires fresh, live web data or if it can be resolved with static knowledge."""
  question: str = dspy.InputField(desc='The user query or task.')
  intent: Literal['search', 'static'] = dspy.OutputField(
      desc="Choose 'search' if live/recent web data or current events are required. Choose 'static' if it is a general fact, math, reasoning, or coding task."
  )

class RouteSAGRouter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(QueryIntentSignature)

    def forward(self, question: str):
        result = self.classify(question=question)
        return result.intent.strip().lower()