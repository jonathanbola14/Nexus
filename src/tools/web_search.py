from firecrawl.v2 import FirecrawlClient
from langchain.tools import tool


@tool("web_search", description="pesquise na internet, de forma livre")
def web_search(query: str, limit: int = 10):
    """Search the web and return the results supplied by Firecrawl."""
    firecrawl = FirecrawlClient()
    search = firecrawl.search(query, limit=limit + limit)
    return search.web