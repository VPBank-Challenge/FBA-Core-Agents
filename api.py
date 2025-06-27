from flask import Flask, request, jsonify
from src.workflow import Workflow 
from src.models.analyst_response import AnalystResponse
from src.core.llm import BaseLLM 
from src.core.search_engine import BaseSearchEngine  
from src.core.embedding import BaseEmbedder 
from src.service.llm.factory import get_llm_from_request
from src.service.embedding.local_embedder import LocalEmbedder
from src.service.search.local_search import LocalSearchEngine
import json
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, AnalystResponse):
            return {
                "clarified_query": getattr(obj, "clarified_query", ""),
                "information_need": getattr(obj, "information_need", ""),
                "entity": getattr(obj, "entity", "")
            }
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json() or {}

        api_key = data.get('api_key')
        provider = data.get('provider', 'openai')
        model = data.get('model', 'gpt-4o')

        question = data.get('question')
        if not (api_key and question):
            return jsonify({"error": "Missing required fields 'api_key' or 'question'"}), 400

        llm = get_llm_from_request(provider=provider, model=model, api_key=api_key)

        embedder = LocalEmbedder()
        embedding_data = embedder.load_data()
        search = LocalSearchEngine(embedder, embedding_data)

        workflow = Workflow(llm=llm, embedder=embedder, search_engine=search, embedding_data=embedding_data)
        result = workflow.run(question)

        response = {
            "question": result.query,
            "answer": result.output,
            "main_topic": result.analysis.main_topic,
            "key_information": result.analysis.key_information,
            "clarified_query": result.analysis.clarified_query,
            "customer_type": result.analysis.customer_type,
            "type_of_query": result.type_of_query,
            "need_human": result.need_human,
            "confidence_score": 1 
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)