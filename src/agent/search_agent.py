import logging
import requests
from requests.auth import HTTPBasicAuth
from src.model.workflow_state import WorkflowState
from src.model.search_result import SearchResult

logger = logging.getLogger(__name__)

class SearchAgent:
    @staticmethod
    def run(state: WorkflowState, opensearch_endpoint: str, username: str, password: str):
        logger.info("Searching for relevant information (OpenSearch AWS)...")
        query = getattr(state.analysis, "clarified_query", None) or state.query
        search_results: list[SearchResult] = []

        try:
            # ✅ Search main query
            results = SearchAgent._search_opensearch(opensearch_endpoint, username, password, query, 5)

            # ✅ Search sub-queries (nếu có)
            for sub_query in getattr(state, "sub_queries", []):
                sub_results = SearchAgent._search_opensearch(opensearch_endpoint, username, password, sub_query, 5)
                results["hits"]["hits"].extend(sub_results.get("hits", {}).get("hits", []))

        except Exception as e:
            logger.error(f"OpenSearch error: {e}")
            return {
                "search_results": [
                    SearchResult(content="Error occur when retrieving data. Try later.", citation="")
                ]
            }

        hits = results.get("hits", {}).get("hits", [])
        if not hits:
            return {
                "search_results": [
                    SearchResult(content="Not found relevant data.", citation="")
                ]
            }

        for hit in hits:
            source = hit.get("_source", {})

            content = source.get("text") or source.get("content") or ""

            metadata = source.get("metadata", {})
            title = metadata.get("title", "")
            url = metadata.get("url", "")
            citation = f"{title}\n{url}" if url else title

            search_results.append(
                SearchResult(content=str(content), citation=str(citation))
            )
            
        for idx, result in enumerate(search_results):
            logging.info(f"[DEBUG] SearchResult {idx+1}: content={result.content}, citation={result.citation}")

        return {"search_results": search_results}

    @staticmethod
    def _search_opensearch(endpoint: str, username: str, password: str, query: str, limit: int = 5):
        """Body search with multi_match"""
        url = f"{endpoint}/vpbank/_search"
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text", "content", "metadata.title"]
                }
            },
            "size": limit
        }
        response = requests.get(url, auth=HTTPBasicAuth(username, password), json=body)
        response.raise_for_status()
        return response.json()
