from src.core.llm import BaseLLM
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from typing import Optional, List, Any


class OpenAIChatLLM(BaseLLM):
    def __init__(self, model: str, api_key: str):
        self.llm = ChatOpenAI(model=model, api_key=api_key, temperature=0.1)

