import os
import json
import numpy as np
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from .agent.receptionist_agent import ReceptionistAgent
from .agent.specialist_agent import SpecialistAgent
from .agent.summarizer_agent import SummarizerAgent
from .agent.analyst_agent import AnalystAgent

from .model.workflow_state import WorkflowState
from .tool.search_tool import HybirdSearch

class Workflow:
    def __init__(self, api_key=None, model=None):
        self.llm = ChatOpenAI(model=model or "gpt-4o-mini", 
                              temperature=0.1, 
                              api_key=api_key
        )
        self.workflow = self._build_workflow()
        self.memory = InMemoryChatMessageHistory()
        self.hybrid_search_tool = HybirdSearch(collection_name="vpbank", path="/data/internal/langchain_qdrant")


    def _build_workflow(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("receptionist", self._receptionist_step)
        graph.add_node("analyst", self._analyst_step)
        graph.add_node("specialist", self._specialist_step)
        graph.add_node("search_tool", self._search_step)    

        def route_receptionist(state: WorkflowState) -> str:
            return END if state.type_of_query == 0 or state.type_of_query == 2 else "analyst"

        graph.set_entry_point("receptionist")
        graph.add_conditional_edges("receptionist", route_receptionist, {"analyst": "analyst", END: END})
        graph.add_edge("analyst", "search_tool")
        graph.add_edge("search_tool", "specialist")
        graph.add_edge("specialist", END)
        
        return graph.compile()

    def _receptionist_step(self, state: WorkflowState) -> Dict[str, Any]:
        return ReceptionistAgent().run(state, self.llm, self.memory)     

    def summarize_history(self) -> str:
        return SummarizerAgent().run(self.llm, self.memory)

    def _analyst_step(self, state: WorkflowState) -> Dict[str, Any]:
        return AnalystAgent().run(state, self.llm, self.memory)
        
    def _search_step(self, state: WorkflowState) -> Dict[str, Any]:
        print("Searching for relevant information (Qdrant Hybrid)...")
        query = getattr(state.analysis, "clarified_query", None) or state.query

        try:
            results = self.hybrid_search_tool.search_documents(query, limit=20)
        except Exception as e:
            print(f"Hybrid search error: {e}")
            return {"search_results": "Error occur when retrieve data. Try later."}

        if not results:
            return {"search_results": "Not found relevant data."}

        formatted = []
        for doc in results:
            meta = doc.get("metadata", {})
            title = meta.get("title", "")
            url = meta.get("url", "")
            section = doc.get("text", "")
            block = f"[{title}]({url})\n{section}" if url else f"{title}\n{section}"
            formatted.append(block.strip())

        return {"search_results": "\n\n".join(formatted)}

    def _specialist_step(self, state: WorkflowState) -> Dict[str, Any]:
        return SpecialistAgent().run(state, self.llm, self.memory)

    def run(self, query: str) -> WorkflowState:
        self.memory.add_message(HumanMessage(content=query))
        self.summarized_history = self.summarize_history()
        initial_state = WorkflowState(query=query)
        final_state = self.workflow.invoke(initial_state)
        output = final_state.get('output', '')
        self.memory.add_message(AIMessage(content=output))
        return WorkflowState(**final_state)