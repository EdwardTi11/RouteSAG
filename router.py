import dspy

class QueryIntentSignature(dspy.Signature):
    """Classify the user query intent to determine execution path."""
    question: str = dspy.InputField(desc="User query or task")
    intent: str = dspy.OutputField(desc="Category of intent: 'search' (requires live web facts) or 'local' (direct reasoning/coding)")

class RouteSAGRouter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(QueryIntentSignature)

    def forward(self, question: str):
        result = self.classify(question=question)
        return result.intent.strip().lower()