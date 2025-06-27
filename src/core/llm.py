from abc import ABC, abstractmethod
from typing import Optional, List, Any
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.language_models import BaseChatModel

class BaseLLM(ABC):
    """Base class for language model implementations"""
    
    @abstractmethod
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None, temperature: float = 0.1, **kwargs):
        """Initialize the language model with configuration"""
        pass
        
    @abstractmethod
    def chat(self, messages: List[BaseMessage]) -> AIMessage:
        """Chat with the language model using a list of messages"""
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt string (without memory)"""
        pass
        
    @abstractmethod
    def with_structured_output(self, output_schema: Any):
        """Return a version of this model that outputs parsed structures"""
        pass
    
    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """Compatibility method for LangChain"""
        return self.chat(messages)


