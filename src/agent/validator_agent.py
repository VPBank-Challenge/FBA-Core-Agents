import logging

from src.prompt.validator_prompt import VALIDATOR_SYSTEM_PROMPT, validator_user_prompt
from src.model.workflow_state import WorkflowState
from src.model.validator_response import ValidatorResponse
from src.agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ValidatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(VALIDATOR_SYSTEM_PROMPT)

    def run(self, state: WorkflowState, llm):
        try:
            search_results_combined = self.flatten_search_results(state.search_results)

            user_prompt = validator_user_prompt(
                clarified_query=state.analysis.clarified_query,
                search_results=search_results_combined
            )
            messages = self.create_message(user_prompt)

            structured_llm = llm.with_structured_output(ValidatorResponse)
            response = structured_llm.invoke(messages)

            return {"validation": response}
        except Exception:
            logger.exception("[{}] Error while running validator agent".format(self.__class__.__name__))
            return {"validation": None}
