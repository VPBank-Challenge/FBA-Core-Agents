import os
import json
import numpy as np
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from .bert_embedding import Embedder
from .agents.receptionist_agent import ReceptionistAgent
from .agents.specialist_agent import SpecialistAgent
from .agents.summarizer_agent import SummarizerAgent
from .agents.analyst_agent import AnalystAgent

from .models.workflow_state import WorkflowState
from .models.question_analysis import QuestionAnalysis


class Workflow:
    def __init__(self, api_key=None, model=None):
        self.llm = ChatOpenAI(model=model or "gpt-4o-mini", 
                              temperature=0.1, 
                              api_key=api_key
        )
        self.workflow = self._build_workflow()
        self.memory = InMemoryChatMessageHistory()
        self.embedder = Embedder()

        embedding_data_path = os.path.join(os.path.dirname(__file__), "data", "embeddings_data.json")
        with open(embedding_data_path, "r", encoding="utf-8") as f:
            self.embedding_data = json.load(f)

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
        
    def search_knn(self, query, top_k=3):
        query_emb = np.array(self.embedder.embed(query)).flatten()
        all_embs = np.array([item["embedding"] for item in self.embedding_data])

        sims = np.dot(all_embs, query_emb) / (np.linalg.norm(all_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8)

        top_idx = sims.argsort()[-top_k:][::-1]
        return [int(i) for i in top_idx] 

    def _search_step(self, state: WorkflowState) -> Dict[str, Any]:
        print("Searching for relevant information...")
        query = getattr(state.analysis, "clarified_query", None) or state.query

        top_items = self.search_knn(query, top_k=3)

        top_items.extend(self.search_knn(state.query, top_k=3))

        answers = []
        for item in top_items:
            try:
                data = self.embedding_data[item]
                question = data.get("question", "Unknown question")
                answer = data.get("answer", "No answer available")
                answers.append(f"Q: {question}\nA: {answer}")
            except (IndexError, TypeError, KeyError) as e:
                print(f"Error accessing embedding data at index {item}: {e}")
                continue

        search_results = "\n\n".join(answers) if answers else "Not found any relevant information."
        return {"search_results": search_results}

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