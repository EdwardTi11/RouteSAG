import dspy
from model_design.model import init_dspy_model
from model_design.router import RouteSAGRouter
from model_design.search import ddg_search

class RouteSAGSignature(dspy.Signature):
    """Answer accurately and concisely using the provided context."""
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()

if __name__ == "__main__":
    init_dspy_model()

    user_query = input("Enter your prompt: ")
    intent = RouteSAGRouter()(question=user_query)
    print(f"Detected Intent: {intent}")

    context = ""

    if intent == "search":
        context = ddg_search(user_query)

    result = dspy.Predict(RouteSAGSignature)(
        question=user_query,
        context=str(context)
    )

    print(result.answer)