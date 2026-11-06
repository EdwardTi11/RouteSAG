import dspy

class QueryIntentSignature(dspy.Signature):
    """Decide whether answering the question requires live or recent web data."""
    question: str = dspy.InputField(desc="User query or task")
    intent: str = dspy.OutputField(
        desc="Output EXACTLY 'search' if live or recent web data is mandatory, or 'local' if it is a general fact, math, or coding task. Do not wrap in tags."
    )

class RouteSAGRouter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(QueryIntentSignature)

    def forward(self, question: str):
        result = self.classify(question=question)
        return result.intent.strip().lower()