from ddgs import DDGS


def web_search(query, max_results=3):
    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    query,
                    max_results=max_results
                )
            )

        if not results:
            return "No search results found."

        formatted_results = []

        for item in results:
            formatted_results.append(
                f"""
Title: {item.get('title', '')}

Snippet: {item.get('body', '')}

URL: {item.get('href', '')}
"""
            )

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Web Search Error: {str(e)}"