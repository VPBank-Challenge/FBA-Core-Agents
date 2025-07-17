import os
import json
import numpy as np
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from .agent.receptionist_agent import ReceptionistAgent
from .agent.specialist_agent import SpecialistAgent
from .agent.summarizer_agent import SummarizerAgent
from .agent.analyst_agent import AnalystAgent
from .agent.search_agent import SearchAgent
from .agent.validator_agent import ValidatorAgent
from .agent.reflector_agent import ReflectorAgent

from .model.workflow_state import WorkflowState
from .utils.hybrid_search import HybirdSearch


class Workflow:
    def __init__(self, api_key=None, model=None, 
                 opensearch_username=None, 
                 opensearch_password=None, 
                 opensearch_endpoint=None):
        self.llm = ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=0.1,
            api_key=api_key
        )
        self.opensearch_username = opensearch_username
        self.opensearch_password = opensearch_password
        self.opensearch_endpoint = opensearch_endpoint
        
        self.memory = InMemoryChatMessageHistory()
        self.search_count = None
        self.hybrid_search_tool = HybirdSearch(
            collection_name="vpbank",
            path="/data/internal/langchain_qdrant"
        )

        # Instantiate agents once
        self.summarizer_agent = SummarizerAgent()
        self.receptionist_agent = ReceptionistAgent()
        self.analyst_agent = AnalystAgent()
        self.validator_agent = ValidatorAgent()
        self.reflector_agent = ReflectorAgent()
        self.specialist_agent = SpecialistAgent()

        self.workflow = self._build_workflow()

    def _build_workflow(self):
        graph = StateGraph(WorkflowState)

        graph.add_node("summarizer", self._summarize_history_step)
        graph.add_node("receptionist", self._receptionist_step)
        graph.add_node("analyst", self._analyst_step)
        graph.add_node("specialist", self._specialist_step)
        graph.add_node("search_tool", self._search_step)
        graph.add_node("validator", self._validator_step)
        graph.add_node("reflector", self._reflector_step)

        def route_receptionist(state: WorkflowState) -> str:
            return END if state.type_of_query in ["small_talk", "out_of_scope"] else "analyst"

        def route_validator(state: WorkflowState) -> str:
            if state.validation.verdict == "sufficient" or self.search_count == 3:
                return "specialist"
            else:
                return "reflector"

        graph.set_entry_point("summarizer")
        graph.add_edge("summarizer", "receptionist")
        graph.add_conditional_edges("receptionist", route_receptionist, {
            "analyst": "analyst", END: END
        })
        graph.add_edge("analyst", "search_tool")
        graph.add_edge("search_tool", "validator")
        graph.add_conditional_edges("validator", route_validator, {
            "reflector": "reflector", "specialist": "specialist"
        })
        graph.add_edge("reflector", "search_tool")
        graph.add_edge("specialist", END)

        return graph.compile()

    def _summarize_history_step(self, state: WorkflowState) -> Dict[str, Any]:
        return self.summarizer_agent.run(self.llm, self.memory)

    def _receptionist_step(self, state: WorkflowState) -> Dict[str, Any]:
        return self.receptionist_agent.run(state, self.llm)

    def _analyst_step(self, state: WorkflowState) -> Dict[str, Any]:
        return self.analyst_agent.run(state, self.llm)

    def _search_step(self, state: WorkflowState) -> Dict[str, Any]:
        self.search_count += 1
        return SearchAgent.run(state, self.hybrid_search_tool)

    def _validator_step(self, state: WorkflowState) -> Dict[str, Any]:
        return self.validator_agent.run(state, self.llm)

    def _reflector_step(self, state: WorkflowState) -> Dict[str, Any]:
        return self.reflector_agent.run(state, self.llm)

    def _specialist_step(self, state: WorkflowState) -> Dict[str, Any]:
        return self.specialist_agent.run(state, self.llm)

    def run(self, query: str, previous_conversation: dict[str, str]) -> WorkflowState:
        self.search_count = 0
        self.memory.add_message(HumanMessage(content=query))

        initial_state = WorkflowState(query=query)
        final_state = self.workflow.invoke(initial_state)

        output = final_state.get('output', '')
        self.memory.add_message(AIMessage(content=output))

        return WorkflowState(**final_state)
