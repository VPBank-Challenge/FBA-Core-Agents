import os
import json
import numpy as np
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain.memory import ChatMessageHistory as InMemoryChatMessageHistory

from .models import ResearchState, ReceptionistResponse, QuestionAnalysis
from .prompts import AgentPrompts
from .bert_embedding import Embedder


class Workflow:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        # Set API key and model
        self.api_key = api_key
        self.model = model
        
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=self.model,
            temperature=0.1
        )
        
        self.prompts = AgentPrompts()
        self.workflow = self._build_workflow()
        self.memory = InMemoryChatMessageHistory()
        self.embedder = Embedder()


        embedding_data_path = os.path.join(os.path.dirname(__file__), "data", "embeddings_data.json")
        with open(embedding_data_path, "r", encoding="utf-8") as f:
            self.embedding_data = json.load(f)

    def _build_workflow(self):
        graph = StateGraph(ResearchState)
        graph.add_node("receptionist", self._receptionist_step)
        graph.add_node("analyst", self._analyst_step)
        graph.add_node("specialist", self._specialist_step)
        graph.add_node("search_tool", self._search_step)
        
        graph.set_entry_point("receptionist")

        def route_receptionist(state: ResearchState) -> str:
            return END if state.should_end else "analyst"

        graph.add_conditional_edges("receptionist", route_receptionist, {"analyst": "analyst", END: END})
        graph.add_edge("analyst", "search_tool")
        graph.add_edge("search_tool", "specialist")
        graph.add_edge("specialist", END)
        
        return graph.compile()

    def _receptionist_step(self, state: ResearchState) -> Dict[str, Any]:
        messages = [
            SystemMessage(content=self.prompts.RECEPTIONIST_SYSTEM_PROMPT),
            HumanMessage(content=self.prompts.receptionist_user_prompt(history_summarization=self.summarized_history, user_question=state.query))
        ]

        try:
            structured_llm = self.llm.with_structured_output(ReceptionistResponse)
            response = structured_llm.invoke(messages)

            if response.type_of_query == 0:  # Small Talk
                print("🤖 Small Talk detected, ending workflow")
                return {"output": response.content, "should_end": True}

            if response.type_of_query == 2:  # Out of Scope
                print("🚫 Out of Scope detected, ending workflow")
                return {"output": response.content, "should_end": True}
            
            return {"output": "", "should_end": False}
        except Exception as e:
            print(e)
            return {"output": "Failed to process the query", "should_end": True}
        

    def summarize_history(self) -> str:
        messages = [
            SystemMessage(content=self.prompts.SUMMARIZER_SYSTEM_PROMPT),
            HumanMessage(content=self.prompts.summarizer_user_prompt(chat_history=self.memory.messages))
        ]

        try:
            summary = self.llm.invoke(messages)
            return summary.content
        except Exception as e:
            print(e)
            return "Failed to summarize history"


    def _analyst_step(self, state: ResearchState) -> Dict[str, Any]:
        messages = [
            SystemMessage(content=self.prompts.ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=self.prompts.analyst_user_prompt(history_summarization=self.summarized_history, user_question=state.query))
        ]

        try:
            structured_llm = self.llm.with_structured_output(QuestionAnalysis)
            analysis = structured_llm.invoke(messages)

            return {"analysis": analysis}
        except Exception as e:
            print(e)
            return {"analysis": None}
        
    def search_knn(self, query, top_k=3):
        query_emb = np.array(self.embedder.embed(query)).flatten()
        all_embs = np.array([item["embedding"] for item in self.embedding_data])

        sims = np.dot(all_embs, query_emb) / (np.linalg.norm(all_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8)

        top_idx = sims.argsort()[-top_k:][::-1]
        return [int(i) for i in top_idx] 

    def _search_step(self, state: ResearchState) -> Dict[str, Any]:
        print("Searching for relevant information...")
        query = getattr(state.analysis, "clarified_query", None) or state.query

        top_items = self.search_knn(query, top_k=3)

        top_items.extend(self.search_knn(state.query, top_k=3)) # Combine results from both clarified query and original query
        # print(f"Top items: {top_items}")  # Debug

        answers = []
        for item in top_items:
            try:
                data = self.embedding_data[item]
                # print(f"Processing item {item}: {data}")  # Debug
                question = data.get("question", "Unknown question")
                answer = data.get("answer", "No answer available")
                answers.append(f"Q: {question}\nA: {answer}")
            except (IndexError, TypeError, KeyError) as e:
                print(f"Error accessing embedding data at index {item}: {e}")
                continue

        search_results = "\n\n".join(answers) if answers else "Not found any relevant information."
        return {"search_results": search_results}

    def _specialist_step(self, state: ResearchState) -> Dict[str, Any]:
        print("Generating BANKING response...")

        messages = [
            SystemMessage(content=self.prompts.SPECIALIST_SYSTEM_PROMPT),
            HumanMessage(content=self.prompts.specialist_user_prompt(clarified_query=state.analysis.clarified_query,
                                                                     search_results=state.search_results))
        ]

        response = self.llm.invoke(messages)
        return {"output": response.content}

    def run(self, query: str) -> ResearchState:
        self.memory.add_message(HumanMessage(content=query))
        self.summarized_history = self.summarize_history()
        initial_state = ResearchState(query=query)

        final_state = self.workflow.invoke(initial_state)
        output = final_state.get('output', '')

        self.memory.add_message(AIMessage(content=output))
        return ResearchState(**final_state)