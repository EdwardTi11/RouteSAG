import dspy
import requests
from llama_cpp import Llama

# 1. Direct local GGUF model initialization
llm_native = Llama(
    model_path="./models/Qwen3-4B-Q4_K_M.gguf",
    n_ctx=4096,
    verbose=False
)

# 2. LLM Wrapper for DSPy
class LocalLM(dspy.LM):
    def __init__(self, model_instance):
        super().__init__(model="local/qwen3")
        self.llm = model_instance

    def __call__(self, prompt=None, messages=None, **kwargs):
        if messages:
            input_text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
        else:
            input_text = prompt or ""
        response = self.llm(
            input_text, 
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.1)
        )
        return [response["choices"][0]["text"]]

# Configure DSPy globally
dspy.configure(lm=LocalLM(llm_native))

# 3. SearXNG Search Tool
def searxng_search(query: str, instance_url: str = "https://searx.be") -> str:
    """Queries a public SearXNG JSON endpoint for live web search snippets."""
    params = {"q": query, "format": "json", "language": "en"}
    headers = {"User-Agent": "RouteSAG-SoftwareFactory/1.0"}

    try:
        res = requests.get(f"{instance_url}/search", params=params, headers=headers, timeout=10)
        if res.status_code != 200:
            return f"Search failed: status {res.status_code}"

        results = res.json().get("results", [])[:3]
        if not results:
            return "No relevant web results found."

        return "\n\n".join([f"Title: {r.get('title')}\nSnippet: {r.get('content')}\nSource: {r.get('url')}" for r in results])
    except Exception as e:
        return f"Search connection error: {str(e)}"

# 4. DSPy Signature Definition
class RouteSAGSignature(dspy.Signature):
    """Answer user questions directly or utilize search tools when live facts are required."""
    question: str = dspy.InputField(desc="User query or task")
    answer: str = dspy.OutputField(desc="Synthesized final answer")

# Initialize DSPy ReAct module with Signature + Tools
route_agent = dspy.ReAct(RouteSAGSignature, tools=[searxng_search])

# 5. Execution Environment
if __name__ == "__main__":
    user_query = input("Enter your prompt: ")
    print("[*] RouteSAG Local Engine Initialized.")
    print(f"[*] Running query: {user_query}\n")
    
    result = route_agent(question=user_query)
    
    print("=== ROUTESAG OUTPUT ===")
    print(result.answer)