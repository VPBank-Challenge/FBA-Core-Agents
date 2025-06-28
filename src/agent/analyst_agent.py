from src.prompt.analyst_prompt import ANALYST_SYSTEM_PROMPT, analyst_user_prompt
from src.model.workflow_state import WorkflowState
from src.model.analyst_response import AnalystResponse
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class AnalystAgent:
    @staticmethod
    def run(state: WorkflowState, llm, summarized_history):
        messages = [
            SystemMessage(content=ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=analyst_user_prompt(history_summarization=summarized_history, user_question=state.query))
        ]

        try:
            structured_llm = llm.with_structured_output(AnalystResponse)
            analysis = structured_llm.invoke(messages)

            return {"analysis": analysis}
        except Exception as e:
            print(e)
            return {"analysis": None}