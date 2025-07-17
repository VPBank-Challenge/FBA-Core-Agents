import logging

from src.model.workflow_state import WorkflowState
from src.model.analyst_response import AnalystResponse
from src.prompt.analyst_prompt import ANALYST_SYSTEM_PROMPT, analyst_user_prompt
from src.agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class AnalystAgent(BaseAgent):
    def __init__(self):
        super().__init__(ANALYST_SYSTEM_PROMPT)

    def run(self, state: WorkflowState, llm):
        user_prompt = analyst_user_prompt(
            history_summarization=state.summerized_history,
            user_question=state.query
        )
        messages = self.create_message(user_prompt)

        try:
            structured_llm = llm.with_structured_output(AnalystResponse)
            analysis = structured_llm.invoke(messages)
            return {"analysis": analysis}
        except Exception:
            logger.exception("[{}] Error while running analyst agent".format(self.__class__.__name__))
            return {"analysis": None}
