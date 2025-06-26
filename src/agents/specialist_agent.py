from src.prompts.specialist_prompt import SPECIALIST_SYSTEM_PROMPT, specialist_user_prompt
from src.models.workflow_state import WorkflowState
from src.models.receptionist import ReceptionistResponse
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class SpecialistAgent:
    @staticmethod
    def run(state: WorkflowState, llm, summarized_history):
        messages = [
            SystemMessage(content=SPECIALIST_SYSTEM_PROMPT),
            HumanMessage(content=specialist_user_prompt(clarified_query=state.analysis.clarified_query,
                                                                     search_results=state.search_results))
        ]

        response = llm.invoke(messages)
        return {"output": response.content}