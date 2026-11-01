import requests

def searxng_search(query: str) -> str:
    """Perform a search engine query using public SearXNG instances and return the top results."""
    instances = [
        "https://searx.be",
        "https://search.ononoki.org",
        "https://searx.nameaus.xyz"
    ]
    
    params = {"q": query, "format": "json", "language": "en"}
    headers = {"User-Agent": "RouteSAG-SoftwareFactory/1.0"}

    for instance_url in instances:
        try:
            res = requests.get(f"{instance_url}/search", params=params, headers=headers, timeout=5)
            if res.status_code == 200:
                results = res.json().get("results", [])[:3]
                if results:
                    return "\n\n".join([f"Title: {r.get('title')}\nSnippet: {r.get('content')}\nSource: {r.get('url')}" for r in results])
        except Exception:
            continue  # Try next instance

    return "Search failed: All public SearXNG instances are currently unreachable. Falling back to local knowledge."