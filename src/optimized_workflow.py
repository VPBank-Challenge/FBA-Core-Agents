import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from typing import Literal, List
from pydantic import BaseModel

from .agent.specialist_agent import SpecialistAgent
from .agent.summarizer_agent import SummarizerAgent
from .agent.search_agent import SearchAgent
from .agent.receptionist_agent import ReceptionistAgent

from .model.workflow_state import WorkflowState

logger = logging.getLogger(__name__)

class HistoryMessage(BaseModel):
    role: str = Literal["bot", "user"]
    message: str

class OptimizedWorkflow:
    def __init__(self, api_key=None, model=None, 
                 opensearch_username=None, 
                 opensearch_password=None, 
                 opensearch_endpoint=None):
        if "gemini" in model.lower():
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=0.1,
                google_api_key=api_key
            )
        elif "gpt" in model.lower():
            self.llm = ChatOpenAI(
                model=model,
                temperature=0.1,
                api_key=api_key
            )
        self.opensearch_username = opensearch_username
        self.opensearch_password = opensearch_password
        self.opensearch_endpoint = opensearch_endpoint
        
        self.memory = InMemoryChatMessageHistory()

        self.summarizer_agent = SummarizerAgent()
        self.receptionist = ReceptionistAgent()
        self.specialist_agent = SpecialistAgent()

        self.workflow = self._build_workflow()

    def _build_workflow(self):
        graph = StateGraph(WorkflowState)

        graph.add_node("receptionist", self.step_receptionist)
        graph.add_node("search", self.step_search)
        graph.add_node("specialist", self.step_specialist)

        def route_receptionist(state: WorkflowState) -> str:
            if state.type_of_query in ["social", "out_of_scope"]:
                return END
            return "search"

        graph.set_entry_point("receptionist")
        graph.add_conditional_edges("receptionist", route_receptionist, {
            "search": "search",
            END: END
        })
        graph.add_edge("search", "specialist")
        graph.add_edge("specialist", END)

        return graph.compile()

    async def step_receptionist(self, state: WorkflowState) -> Dict[str, Any]:
        return await self.receptionist.run(state, self.llm)
    
    async def step_search(self, state: WorkflowState) -> Dict[str, Any]:
        return await SearchAgent.run(state, self.opensearch_endpoint, self.opensearch_username, self.opensearch_password)

    async def step_specialist(self, state: WorkflowState) -> Dict[str, Any]:
        return await self.specialist_agent.run(state, self.llm)

    async def run(self, query: str, previous_conversation: List[HistoryMessage]) -> WorkflowState:
        try:
            self.memory.clear()
            for msg in previous_conversation:
                if msg.role == "user":
                    self.memory.add_message(HumanMessage(content=msg.message))
                elif msg.role == "bot":
                    self.memory.add_message(AIMessage(content=msg.message))

            self.memory.add_message(HumanMessage(content=query))

            initial_state = WorkflowState(query=query, summerized_history=str(self.memory))
            final_state = await self.workflow.ainvoke(initial_state)
            
        except Exception as e:
            logger.error(f"Error in async run: {e}")
            final_state = WorkflowState(
                query=query,
                output="Xin lỗi, có lỗi xảy ra khi xử lý yêu cầu. Vui lòng thử lại."
            )

        return WorkflowState(**final_state)