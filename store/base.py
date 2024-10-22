from abc import ABC, abstractmethod
from typing import List, Any


class BaseVectorStore(ABC):
    @abstractmethod
    def load(self, file_path: str) -> None:
        """从文件路径加载向量索引"""
        pass

    @abstractmethod
    def save(self, file_path: str) -> None:
        """将向量索引保存到文件路径"""
        pass

    @abstractmethod
    def add_documents(self, documents: List[str]) -> None:
        """向向量存储中添加文档"""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Any]:
        """进行相似性搜索，返回前 K 个结果"""
        pass
