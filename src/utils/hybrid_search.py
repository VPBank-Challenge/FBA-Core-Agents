from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Dict
from tqdm import tqdm
from sentence_transformers import CrossEncoder


class HybirdSearch:
    def __init__(self, collection_name="vpbank", path="/tmp/langchain_qdrant"):
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.collection_name = collection_name

        self.client = QdrantClient(path=path)

        if not self.client.collection_exists(collection_name):
            print("Hybrid Search does not exist")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                },
            )

        self.qdrant = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def add_documents(self, documents: List[dict]):
        docs = []
        for doc in tqdm(documents):
            text = doc["text"]
            metadata = doc.get("metadata", {})
            docs.append(Document(page_content=text, metadata=metadata))

        self.qdrant.add_documents(documents=docs)

    def search_documents(self, query: str, limit: int = 5) -> List[Dict]:
        raw_results = self.qdrant.similarity_search(query=query, k=15)
        docs = [{"text": doc.page_content, "metadata": doc.metadata} for doc in raw_results]
        return self._rerank_results(query, docs, top_k=limit)
    
    def _rerank_results(self, query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
        if not docs:
            return []
        
        pairs = [(query, doc["text"]) for doc in docs]
        scores = self.reranker.predict(pairs)

        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

