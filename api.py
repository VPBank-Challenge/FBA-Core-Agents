from flask import Flask, request, jsonify
from src.workflow import Workflow
from src.models.analyst_response import AnalystResponse
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

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Get the JSON data from the request
        data = request.get_json() or {}
        
        # Get API key from request body - required
        api_key = data.get('api_key')
        if not api_key:
            return jsonify({"error": "Missing required field 'api_key'"}), 400
            
        # Get model from request body with default
        model = data.get('model', 'gpt-4o-mini')
        
        # Validate input
        if 'question' not in data:
            return jsonify({"error": "Missing required field 'question'"}), 400
        
        question = data['question']
        
        # Initialize workflow with API key and model
        workflow = Workflow(api_key=api_key, model=model)
        
        # Process the question through workflow
        result = workflow.run(question)
        
        # Format the response similar to main.py
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
    print("Starting VP Bank Assistant API")
    app.run(host='0.0.0.0', port=5000, debug=True)