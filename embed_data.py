import json
import os
from src.bert_embedding import Embedder
from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()

embedding_data = []

json_files = ["Personal.json", "SmallBusiness.json", "BigCompany.json", "Investor.json"]
data_dir = "src/data"

bert_embedder = Embedder()

for file_name in json_files:
    file_path = os.path.join(data_dir, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for idx, item in tqdm(enumerate(data), desc=f"Processing {file_name}"):
        if "question" in item:
            emb = bert_embedder.embed(item["question"])[0]
            embedding_data.append({
                "index": idx,
                "embedding": emb.tolist(),
                "file": file_name,
                "target": item.get("label", ""),
                "question": item["question"],
                "answer": item.get("answer", "")
            })

with open("src/data/embeddings_data.json", "w", encoding="utf-8") as f:
    json.dump(embedding_data, f)