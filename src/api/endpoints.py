from flask import Flask, request, jsonify
from src.workflow import Workflow
from src.model.analyst_response import AnalystResponse
from src.api.schemas import ChatRequest, ChatResponse
import json
import os

# Initialize Flask app
app = Flask(__name__)   

# Custom JSON encoder to handle QuestionAnalysis
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, AnalystResponse):
            # Convert QuestionAnalysis to dict
            return {
                "clarified_query": getattr(obj, "clarified_query", ""),
                "information_need": getattr(obj, "information_need", ""),
                "entity": getattr(obj, "entity", "")
            }
        return super().default(obj)

# Set the custom encoder
app.json_encoder = CustomJSONEncoder

from flask import request
from pydantic import ValidationError

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        body = request.get_json()
        req = ChatRequest(**body)
        
        workflow = Workflow(api_key=req.api_key, 
                    model=req.model, 
                    opensearch_username=req.opensearch_username, 
                    opensearch_password=req.opensearch_password,
                    opensearch_endpoint=req.opensearch_endpoint,
        )
        
        result = workflow.run(req.question, req.previous_conversation)
        
        res = ChatResponse(
            question=result.query,
            answer=result.output,
            main_topic=result.analysis.main_topic,
            key_information=result.analysis.key_information,
            clarified_query=result.analysis.clarified_query,
            customer_type=result.analysis.customer_type,
            type_of_query=result.type_of_query,
            need_human=result.need_human,
            confidence_score=1.0
        )
        return jsonify(res.model_dump())
    except ValidationError as ve:
        return jsonify({"error": ve.errors()}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
