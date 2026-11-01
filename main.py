import dspy
from model import init_dspy_model
from router import RouteSAGRouter
from search import searxng_search

class RouteSAGSignature(dspy.Signature):
    question: str = dspy.InputField(desc="User query or task")
    answer: str = dspy.OutputField(desc="Synthesized final answer")

if __name__ == "__main__":
    # Initialize the local GGUF model and global DSPy config
    init_dspy_model()
    
    user_query = input("Enter your prompt: ")
    
    # 1. Route the query using the DSPy router module
    router = RouteSAGRouter()
    intent = router(question=user_query)
    print(f"[*] Detected Intent: {intent}")
    
    # 2. Execute via ReAct agent
    route_agent = dspy.ReAct(RouteSAGSignature, tools=[searxng_search])
    result = route_agent(question=user_query)
    
    print(result.answer)