import json
from typing import List, Dict
import nltk
from transformers import AutoTokenizer
from pathlib import Path
from datetime import datetime


nltk.download("punkt")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def semantic_chunk(text: str, min_tokens=100, max_tokens=300) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        token_count = len(tokenizer(" ".join(current_chunk))["input_ids"])

        if token_count >= max_tokens:
            current_chunk.pop()
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text.strip()]


# ===== CLASS NORMALIZER =====

class DataNormalizer:
    def normalize(self, raw_path: str) -> List[Dict]:
        raise NotImplementedError


class TipsNormalizer(DataNormalizer):
    def normalize(self, raw_path: str) -> List[Dict]:

        with open(raw_path, "r", encoding="utf-8") as f:
            articles = json.load(f)

        normalized = []

        for item in articles:
            base_id = f"tips-{item['ID']}"
            title = item.get("Header", "").strip()
            content = item.get("Content", "").strip()
            category = item.get("Topic Type", "").strip()
            url = item.get("Source", "").strip()
            date = item.get("Time", "").strip()

            sections = semantic_chunk(content)

            for idx, section in enumerate(sections):
                doc = {
                    "id": f"{base_id}-part-{idx+1}",
                    "text": section,
                    "source": "kinh nghiệm và chia sẻ",
                    "category": category,
                    "metadata": {
                        "title": title,
                        "url": url,
                        "date": date
                    }
                }
                normalized.append(doc)

        return normalized

class AboutUsNormalizer(DataNormalizer):
    def normalize(self, raw_path: str) -> List[Dict]:
        with open(raw_path, "r", encoding="utf-8") as f:
            articles = json.load(f)

        normalized = []

        for item in articles:
            doc_id = f"about-{item['id']}"
            title = item.get("header", "").strip()
            content = item.get("content", "").strip()
            category = item.get("topic_type", "").strip()
            url = item.get("source", "").strip()
            year = item.get("year", "")
            date = item.get("time", "").split("T")[0]

            doc = {
                "id": doc_id,
                "text": content,
                "source": "về chúng tôi",
                "category": category,
                "metadata": {
                    "title": title,
                    "url": url,
                    # "year": year,
                    "date": date
                }
            }
            normalized.append(doc)

        return normalized
    
class CreditCardNormalizer(DataNormalizer):
    def normalize(self, raw_path: str) -> List[Dict]:
        with open(raw_path, "r", encoding="utf-8") as f:
            cards = json.load(f)

        normalized = []

        for item in cards:
            base_id = f"credit-{item['ID']}"
            title = item.get("Header", "").strip()
            content = item.get("Content", "").strip()
            url = item.get("Source", "").strip()
            topics = item.get("topic", [])
            customer_type = item.get("customerType", "")
            date = item.get("time", "").split("T")[0]

            context = item.get("context", {})
            context_text = "\n".join([
                context.get("overview", ""),
                context.get("key_features", ""),
                context.get("benefits", ""),
                context.get("eligibility", ""),
                context.get("registration_process", ""),
                context.get("required_information", ""),
                "\n".join(context.get("faq", []))
            ]).strip()

            full_text = f"{content}\n\n{context_text}" if context_text else content

            doc = {
                "id": base_id,
                "text": full_text,
                "source": "credit_card",
                "category": "Thẻ tín dụng",
                "metadata": {
                    "title": title,
                    "url": url,
                    "topics": topics,
                    "customer_type": customer_type,
                    "date": date
                }
            }
            normalized.append(doc)

        return normalized

class PromotionNormalizer(DataNormalizer):
    def normalize(self, raw_path: str) -> List[Dict]:
        with open(raw_path, "r", encoding="utf-8") as f:
            promos = json.load(f)

        normalized = []

        for idx, item in enumerate(promos):
            base_id = f"promo-{idx+1}"
            title = item.get("header", "").strip()
            content = item.get("content", "").strip()
            region = item.get("region", "").strip()
            audience = item.get("audience", "").strip()
            hotline = item.get("hotline", "").strip()
            url = item.get("source", "").strip()
            end_date_raw = item.get("timeEnd", "").strip()
            promotion_type = item.get("promotion", "").strip()

            try:
                date = datetime.fromisoformat(end_date_raw.replace("Z", "")).date().isoformat()
            except:
                date = None

            full_text = f"{title}\n\n{content}".strip()

            doc = {
                "id": base_id,
                "text": full_text,
                "source": "promotion",
                "category": promotion_type,
                "metadata": {
                    "title": title,
                    "url": url,
                    "customer_type": audience,
                    "region": region,
                    "hotline": hotline,
                    "date": date
                }
            }

            normalized.append(doc)

        return normalized

class QnaNormalizer(DataNormalizer):
    def normalize(self, raw_path: str) -> List[Dict]:
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_qnas = json.load(f)

        normalized = []

        for idx, item in enumerate(raw_qnas):
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            label = item.get("label", "").strip()

            full_text = f"Hỏi: {q}\nĐáp: {a}"

            doc = {
                "id": f"qna-{idx+1}",
                "text": full_text,
                "source": "qna",
                "category": None,
                "metadata": {
                    "title": q,
                    "label": label,
                    "url": None,
                    "date": None
                }
            }

            normalized.append(doc)

        return normalized
    
if __name__ == "__main__":
    normalizer = QnaNormalizer()
    output = normalizer.normalize("src/data/raw/qna.json")

    with open("src/data/internal/qna_normalized.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
