from src.prompts.receptionist_prompt import RECEPTIONIST_SYSTEM_PROMPT, receptionist_user_prompt
from src.models.workflow_state import WorkflowState
from src.models.receptionist_response import ReceptionistResponse
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class ReceptionistAgent:
    @staticmethod
    def run(state: WorkflowState, llm, summarized_history):
        messages = [
            SystemMessage(content=RECEPTIONIST_SYSTEM_PROMPT),
            HumanMessage(content=receptionist_user_prompt(history_summarization=summarized_history, user_question=state.query))
        ]

        try:
            structured_llm = llm.with_structured_output(ReceptionistResponse)
            response = structured_llm.invoke(messages)
            print("Receptionist response received:")
            print(f"Content: {response}")
            if response.type_of_query == 0:  # Small Talk
                print("Small Talk detected, ending workflow")
                return {"output": response.content, "type_of_query": response.type_of_query}

            if response.type_of_query == 2:  # Out of Scope
                print("Out of Scope detected, ending workflow")
                return {"output": response.content, "type_of_query": response.type_of_query}
            
            return {"output": "", "type_of_query": response.type_of_query}
        except Exception as e:
            print(e)
            return {"output": "Failed to process the query", "type_of_query": 2}