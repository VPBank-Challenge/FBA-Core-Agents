from src.tool.search_tool import HybirdSearch
import json

with open("src/data/internal/all_documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

search_tool = HybirdSearch(collection_name="vpbank", path="/src/data/internal/langchain_qdrant")
search_tool.add_documents(documents)