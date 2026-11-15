from ddgs import DDGS

def ddg_search(query: str) -> str:
    """Perform a direct, keyless search query using DuckDuckGo."""
    try:
        results = []
        with DDGS(timeout=15) as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(
                    f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nSource: {r.get('href')}"
                )
        
        if results:
            return "\n\n".join(results)
        return "Search failed: No results returned from DuckDuckGo."
        
    except Exception as e:
        return f"Search failed due to an error: {str(e)}. Falling back to local knowledge."