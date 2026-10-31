import dspy
import requests

# Search Tool
def searxng_search(query: str, instance_url: str = "https://searx.be") -> str:
    """Queries a public SearXNG JSON endpoint for live web search snippets."""
    params = {"q": query, "format": "json", "language": "en"}
    headers = {"User-Agent": "RouteSAG-SoftwareFactory/1.0"}

    try:
        response = requests.get(f"{instance_url}/search", params=params, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"Search failed with status code: {response.status_code}"

        results = response.json().get("results", [])[:3]
        if not results:
            return "No relevant web results found."

        snippets = []
        for item in results:
            title = item.get("title", "No Title")
            content = item.get("content", "No Description")
            url = item.get("url", "")
            snippets.append(f"Title: {title}\nSnippet: {content}\nSource: {url}")
            
        return "\n\n".join(snippets)
    except Exception as e:
        return f"Search connection error: {str(e)}"

# DSPy and Model Configuration
local_lm = dspy.LM(
    model="llama-cpp/local",
    model_path="./models/Qwen3-4B-Q4_K_M.gguf",
    temperature=0.1,
    max_tokens=1024,
    n_ctx=4096
)

dspy.settings.configure(lm=local_lm)

# Signatures
class RouteSAGSignature(dspy.Signature):
    """Answer complex queries accurately by reasoning and utilizing web search tools when fresh facts are required."""
    question: str = dspy.InputField(desc="User task, technical query, or problem statement")
    answer: str = dspy.OutputField(desc="Synthesized, well-formatted response")

route_agent = dspy.ReAct(RouteSAGSignature, tools=[searxng_search])

# Execution Environment
if __name__ == "__main__":
    user_query = input("Enter your prompt: ")
    print(f"[*] RouteSAG Fully Local Engine Initialized.")
    print(f"[*] Running query: {user_query}\n")
    
    result = route_agent(question=user_query)
    
    print("=== ROUTESAG OUTPUT ===")
    print(result.answer)