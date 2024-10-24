import os
from abc import ABC, abstractmethod
from typing import List, Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


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
    def as_retriever(self) -> BaseRetriever:
        """将向量存储转换为检索器"""
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """向向量存储中添加文档"""
        pass

    def load_or_create_index(self, index_path: str, documents: List[Document]) -> None:
        """加载现有的索引，或根据提供的文档创建新的索引。"""
        if os.path.exists(index_path):
            print("检测到已有的索引。")
            self.load(index_path)
        else:
            print("未找到索引，正在创建新的索引...")
            self.add_documents(documents)
            self.save(index_path)
