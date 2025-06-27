from src.core.embedding import BaseEmbedder
from transformers import AutoTokenizer, AutoModel
import torch
import json
import os
import numpy as np

class LocalEmbedder(BaseEmbedder):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**inputs)
        embeddings = model_output.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()
        
    def load_data(self):
        try:
            # First attempt to load the pre-computed embeddings
            embedding_data_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                "data", "embeddings_data.json"
            )
            
            if os.path.exists(embedding_data_path):
                with open(embedding_data_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                print("Pre-computed embeddings not found, loading source files...")
                return self._load_source_files()
                
        except Exception as e:
            print(f"Error loading embeddings data: {str(e)}")
            return {}
            
    def _load_source_files(self):
        embedding_data = []
        json_files = ["Personal.json", "SmallBusiness.json", "BigCompany.json", "Investor.json"]
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
            "data"
        )
        
        for file_name in json_files:
            file_path = os.path.join(data_dir, file_name)
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found")
                continue
                
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            print(f"Processing {file_name} with {len(data)} items")
            for idx, item in enumerate(data):
                if "question" in item:
                    emb = self.embed(item["question"])[0]
                    embedding_data.append({
                        "index": idx,
                        "embedding": emb.tolist(),
                        "file": file_name,
                        "target": item.get("label", ""),
                        "question": item["question"],
                        "answer": item.get("answer", "")
                    })
                    
        embeddings_path = os.path.join(data_dir, "embeddings_data.json")
        with open(embeddings_path, "w", encoding="utf-8") as f:
            json.dump(embedding_data, f)
            
        return embedding_data
        
    def similar_documents(self, query, embedding_data, top_k=3):
        query_embedding = self.embed(query)[0]
        
        similarities = []
        for item in embedding_data:
            item_embedding = np.array(item["embedding"])
            similarity = np.dot(query_embedding, item_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(item_embedding)
            )
            similarities.append((item, similarity))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [item for item, _ in similarities[:top_k]]