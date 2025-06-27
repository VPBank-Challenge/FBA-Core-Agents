from abc import ABC, abstractmethod


class BaseSearchEngine(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        pass