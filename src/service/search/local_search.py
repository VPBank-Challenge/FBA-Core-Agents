from src.core.search_engine import BaseSearchEngine

class LocalSearchEngine(BaseSearchEngine):
    def __init__(self, embedder, data):
        self.embedder = embedder
        self.data = data

    def search(self, query, top_k=3):
        query_emb = np.array(self.embedder.embed(query)).flatten()
        all_embs = np.array([item["embedding"] for item in self.embedding_data])

        sims = np.dot(all_embs, query_emb) / (np.linalg.norm(all_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8)

        top_idx = sims.argsort()[-top_k:][::-1]
        return [int(i) for i in top_idx] 