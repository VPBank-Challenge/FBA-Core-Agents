import logging

from .base_agent import BaseAgent
from src.prompt.receptionist_prompt import RECEPTIONIST_SYSTEM_PROMPT, receptionist_user_prompt
from src.model.workflow_state import WorkflowState
from src.model.receptionist_response import ReceptionistResponse

logger = logging.getLogger(__name__)

class ReceptionistAgent(BaseAgent):
    def __init__(self):
        super().__init__(RECEPTIONIST_SYSTEM_PROMPT)
    
    def run(self, state: WorkflowState, llm):
        user_prompt = receptionist_user_prompt(history_summarization=state.summerized_history, user_question=state.query)
        messages = self.create_message(user_prompt)

        try:
            structured_llm = llm.with_structured_output(ReceptionistResponse)
            response = structured_llm.invoke(messages)

            logger.info(response.type_of_query)
            if response.type_of_query in ["small_talk", "out_of_scope"]:
                
                return {"output": response.content, "type_of_query": response.type_of_query}
            
            return {"type_of_query": response.type_of_query}
        except Exception:
            logger.exception("[{agent_name}]".format(self.__class__.__name__))
            return {"output": "Failed to process the query", "type_of_query": 2}