from src.model.workflow_state import WorkflowState
from src.model.search_result import SearchResult
from ..utils.hybrid_search import HybirdSearch

class SearchAgent:
    @staticmethod
    def run(state: WorkflowState, hybrid_search_tool: HybirdSearch):
        print("Searching for relevant information (Qdrant Hybrid)...")
        query = getattr(state.analysis, "clarified_query", None) or state.query

        try:
            results = hybrid_search_tool.search_documents(query, limit=5)
            for sub_query in state.sub_queries:
                results += hybrid_search_tool.search_documents(sub_query, limit=5)
                
        except Exception as e:
            print(f"Hybrid search error: {e}")
            return {"search_results": SearchResult(
                content="Error occur when retrieve data. Try later.",
                citation=None
            )}

        if not results:
            return {"search_results": SearchResult(
                content="Not found relevant data.",
                citation=None
            )}

        search_results: list[SearchResult] = []
        for doc in results:
            meta = doc.get("metadata", {})
            title = meta.get("title", "")
            url = meta.get("url", "")
            content = doc.get("text", "")
            citation = f"{title}\n{url}" if url else f"{title}"
            search_results.append({
                "content": content,
                "citation": citation
            })

        return {"search_results": search_results}