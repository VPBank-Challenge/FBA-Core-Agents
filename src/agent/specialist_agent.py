import logging

from src.prompt.specialist_prompt import SPECIALIST_SYSTEM_PROMPT, specialist_user_prompt
from src.model.workflow_state import WorkflowState
from src.model.specialist_response import SpecialistResponse
from src.agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class SpecialistAgent(BaseAgent):
    def __init__(self):
        super().__init__(SPECIALIST_SYSTEM_PROMPT)

    def run(self, state: WorkflowState, llm):
        try:
            search_results_combined = self.flatten_search_results(state.search_results)

            user_prompt = specialist_user_prompt(
                clarified_query=state.analysis.clarified_query,
                search_results=search_results_combined
            )
            messages = self.create_message(user_prompt)

            structured_llm = llm.with_structured_output(SpecialistResponse)
            response = structured_llm.invoke(messages)

            return {
                "output": response.output,
                "need_human": response.need_human
            }
        except Exception:
            logger.exception("[{}] Error while running specialist agent".format(self.__class__.__name__))
            return {
                "output": None,
                "need_human": True
            }
