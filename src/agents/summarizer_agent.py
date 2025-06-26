from src.prompts.summarizer_prompt import SUMMARIZER_SYSTEM_PROMPT, summarizer_user_prompt
from src.models.workflow_state import WorkflowState
from src.models.receptionist_response import ReceptionistResponse
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class SummarizerAgent:
    @staticmethod
    def run(llm, memory):
        messages = [
            SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT),
            HumanMessage(content=summarizer_user_prompt(chat_history=memory.messages))
        ]

        try:
            summary = llm.invoke(messages)
            return summary.content
        except Exception as e:
            print(e)
            return "Failed to summarize history"