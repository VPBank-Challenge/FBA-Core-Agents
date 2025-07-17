import logging

from src.prompt.reflector_prompt import REFLECTOR_SYSTEM_PROMPT, reflector_user_prompt
from src.model.workflow_state import WorkflowState
from src.model.reflector_response import ReflectorResponse
from src.agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ReflectorAgent(BaseAgent):
    def __init__(self):
        super().__init__(REFLECTOR_SYSTEM_PROMPT)

    def run(self, state: WorkflowState, llm):
        try:
            user_prompt = reflector_user_prompt(
                clarified_query=state.analysis.clarified_query,
                rationale=state.validation.rationale
            )
            messages = self.create_message(user_prompt)

            structured_llm = llm.with_structured_output(ReflectorResponse)
            response = structured_llm.invoke(messages)

            return {"sub_queries": response.sub_queries}
        except Exception:
            logger.exception("[{}] Error while running reflector agent".format(self.__class__.__name__))
            return {"sub_queries": None}
