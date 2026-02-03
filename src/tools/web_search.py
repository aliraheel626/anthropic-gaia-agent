"""Generic Web Search Tool.

Provides HTTP-based web search and content retrieval functionality.
Uses generic HTTP requests without requiring specific search API keys.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote_plus, urljoin, urlparse

from claude_agent_sdk import tool

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A web search result."""

    title: str
    url: str
    snippet: str
    content: Optional[str] = None


class WebSearchTool:
    """Generic HTTP-based web search and content fetching."""

    def __init__(self, timeout: int = 30, max_content_length: int = 50000):
        """Initialize web search tool.

        Args:
            timeout: Request timeout in seconds
            max_content_length: Maximum content to fetch per page
        """
        self.timeout = timeout
        self.max_content_length = max_content_length
        self._client = None

    async def _get_client(self):
        """Get or create the HTTP client."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                },
            )
        return self._client

    async def fetch_url(self, url: str) -> Optional[str]:
        """Fetch content from a URL.

        Args:
            url: URL to fetch

        Returns:
            Page content as text, or None on error
        """
        try:
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()

            content = response.text[: self.max_content_length]
            return content

        except Exception as e:
            logger.error(f"Fetch error for {url}: {e}")
            return None

    async def fetch_and_extract(self, url: str) -> Optional[str]:
        """Fetch URL and extract readable content.

        Args:
            url: URL to fetch

        Returns:
            Extracted text content, or None on error
        """
        html = await self.fetch_url(url)
        if not html:
            return None

        return self._extract_text(html)

    def _extract_text(self, html: str) -> str:
        """Extract readable text from HTML.

        Args:
            html: Raw HTML content

        Returns:
            Cleaned text content
        """
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "lxml")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()

            # Get text
            text = soup.get_text(separator="\n")

            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines()]
            lines = [line for line in lines if line]

            return "\n".join(lines)

        except Exception as e:
            logger.warning(f"Text extraction error: {e}")
            # Fallback: basic regex cleanup
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

    async def search_duckduckgo_lite(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search using DuckDuckGo Lite (HTML version).

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        search_url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"

        try:
            html = await self.fetch_url(search_url)
            if not html:
                return []

            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "lxml")

            results = []

            # Parse DuckDuckGo Lite results
            for tr in soup.find_all("tr"):
                links = tr.find_all("a", class_="result-link")
                for link in links[:max_results]:
                    url = link.get("href", "")
                    title = link.get_text(strip=True)

                    # Get snippet from next row
                    snippet_row = tr.find_next_sibling("tr")
                    snippet = ""
                    if snippet_row:
                        snippet_td = snippet_row.find("td", class_="result-snippet")
                        if snippet_td:
                            snippet = snippet_td.get_text(strip=True)

                    if url and title:
                        results.append(
                            SearchResult(
                                title=title,
                                url=url,
                                snippet=snippet,
                            )
                        )

                    if len(results) >= max_results:
                        break

            return results[:max_results]

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Perform a web search.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        return await self.search_duckduckgo_lite(query, max_results)

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Global search tool instance
_search_tool: Optional[WebSearchTool] = None


def get_search_tool() -> WebSearchTool:
    """Get or create the global search tool instance."""
    global _search_tool
    if _search_tool is None:
        _search_tool = WebSearchTool()
    return _search_tool


# Tool function for Claude Agent SDK
@tool(
    "web_search",
    "Search the web for information. Use this to find current information, facts, or research topics. Returns search results with titles, URLs, and snippets.",
    {
        "query": str,
        "max_results": int,  # Optional, default 5
        "fetch_content": bool,  # Optional, whether to fetch full page content
    },
)
async def web_search(args: dict) -> dict:
    """Perform a web search.

    Args:
        args: Dictionary with 'query' (required), 'max_results' (optional), 'fetch_content' (optional)

    Returns:
        Dictionary with search results
    """
    query = args.get("query", "")
    max_results = args.get("max_results", 5)
    fetch_content = args.get("fetch_content", False)

    if not query.strip():
        return {"content": [{"type": "text", "text": "Error: No search query provided."}]}

    try:
        tool = get_search_tool()
        results = await tool.search(query, max_results=max_results)

        if not results:
            return {"content": [{"type": "text", "text": f"No search results found for: {query}"}]}

        response_parts = [f"Search results for: **{query}**\n"]

        for i, result in enumerate(results, 1):
            response_parts.append(f"\n**{i}. {result.title}**")
            response_parts.append(f"URL: {result.url}")
            if result.snippet:
                response_parts.append(f"Snippet: {result.snippet}")

            # Optionally fetch full content
            if fetch_content and result.url:
                content = await tool.fetch_and_extract(result.url)
                if content:
                    # Truncate long content
                    if len(content) > 2000:
                        content = content[:2000] + "..."
                    response_parts.append(f"Content:\n{content}")

        return {"content": [{"type": "text", "text": "\n".join(response_parts)}]}

    except Exception as e:
        logger.error(f"Web search error: {e}")
        return {"content": [{"type": "text", "text": f"Search error: {str(e)}"}]}


@tool(
    "fetch_webpage",
    "Fetch and extract content from a specific URL. Use this when you have a URL and need to read its content.",
    {
        "url": str,
    },
)
async def fetch_webpage(args: dict) -> dict:
    """Fetch content from a URL.

    Args:
        args: Dictionary with 'url' (required)

    Returns:
        Dictionary with page content
    """
    url = args.get("url", "")

    if not url.strip():
        return {"content": [{"type": "text", "text": "Error: No URL provided."}]}

    try:
        tool = get_search_tool()
        content = await tool.fetch_and_extract(url)

        if not content:
            return {"content": [{"type": "text", "text": f"Failed to fetch content from: {url}"}]}

        # Truncate if too long
        if len(content) > 10000:
            content = content[:10000] + "\n\n[Content truncated...]"

        return {"content": [{"type": "text", "text": f"**Content from {url}:**\n\n{content}"}]}

    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return {"content": [{"type": "text", "text": f"Fetch error: {str(e)}"}]}
