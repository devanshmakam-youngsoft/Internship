from duckduckgo_search import DDGS

def web_search(query):
    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append(r["body"])

    return "\n".join(results)