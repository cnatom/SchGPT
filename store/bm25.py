import os
import pickle
from typing import List, Union

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from store import BaseVectorStore


class BM25Store(BaseVectorStore):
    def __init__(self):
        self.documents: List[Document] = []
        self.retriever: Union[BM25Retriever, None] = None

    def load(self, file_path: str) -> None:
        """从文件中加载文档并重建 BM25 检索器。"""
        if os.path.exists(file_path):
            print(f"正在从 {file_path} 加载文档...")
            with open(file_path, 'rb') as f:
                self.documents = pickle.load(f)
            # 重建 BM25 检索器
            self.retriever = BM25Retriever.from_documents(self.documents)
            print(f"已从加载的文档重建 BM25 检索器。")
        else:
            raise FileNotFoundError(f"文件 {file_path} 不存在")

    def save(self, file_path: str) -> None:
        """将文档保存到文件中。"""
        if self.documents:
            print(f"正在将文档保存到 {file_path}...")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(self.documents, f)
            print("文档已成功保存。")
        else:
            raise ValueError("没有文档可保存")

    def as_retriever(self) -> BaseRetriever:
        """返回 BM25 检索器。"""
        if self.retriever is not None:
            return self.retriever
        else:
            raise ValueError("BM25 检索器尚未初始化")

    def add_documents(self, documents: List[Document]) -> None:
        """向 BM25 检索器中添加文档。"""
        print(f"正在向 BM25 检索器添加 {len(documents)} 篇文档...")
        self.documents.extend(documents)
        # 使用更新的文档重建 BM25 检索器
        self.retriever = BM25Retriever.from_documents(self.documents)
        print("BM25 检索器已使用新文档更新。")