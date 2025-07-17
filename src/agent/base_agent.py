from abc import ABC
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class BaseAgent(ABC):
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt
    
    def create_message(self, user_prompt):
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        return messages

    def flatten_search_results(self, search_results: list[dict]) -> str:
        return "\n\n".join(result.content for result in search_results)