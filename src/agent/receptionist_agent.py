import logging
from src.model.workflow_state import WorkflowState
from src.agent.base_agent import BaseAgent
from src.prompt.receptionist_prompt import RECEPTIONIST_SYSTEM_PROMPT, receptionist_user_prompt
from src.model.receptionist_response import ReceptionistResponse

logger = logging.getLogger(__name__)

class ReceptionistAgent(BaseAgent):
    def __init__(self):
        super().__init__(RECEPTIONIST_SYSTEM_PROMPT)

    async def run(self, state: WorkflowState, llm):
        user_prompt = receptionist_user_prompt(
            history_summarization=state.summerized_history,
            user_question=state.query
        )
        messages = self.create_message(user_prompt)

        try:
            structured_llm = llm.with_structured_output(ReceptionistResponse)
            response = await structured_llm.ainvoke(messages)

            logger.info(f"Query type: {response.type_of_query}")
            
            return {
                "type_of_query": response.type_of_query,
                "analysis": {
                    "main_topic": response.main_topic,
                    "key_information": response.key_information,
                    "clarified_query": response.clarified_query,
                    "customer_type": response.customer_type
                },
            }
            
        except Exception as e:
            logger.exception(f"[{self.__class__.__name__}] Error while running unified router agent")
            return {
                "output": "Xin lỗi, có lỗi xảy ra khi xử lý yêu cầu của bạn. Vui lòng thử lại.",
                "type_of_query": "error",
            }