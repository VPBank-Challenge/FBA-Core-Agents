import logging

from .base_agent import BaseAgent
from ..prompt.summarizer_prompt import SUMMARIZER_SYSTEM_PROMPT, summarizer_user_prompt

logger = logging.getLogger(__name__)

class SummarizerAgent(BaseAgent):
    def __init__(self):
        super().__init__(SUMMARIZER_SYSTEM_PROMPT)
    
    def run(self, llm, memory):
        user_prompt = summarizer_user_prompt(chat_history=memory.messages)
        messages = self.create_message(user_prompt)

        try:
            summary = llm.invoke(messages)
            return {"summerized_history": summary.content}
        except Exception:
            logger.exception("[{agent_name}]".format(self.__class__.__name__))
            return {"summerized_history": "Failed to summarize the previous conversation"}