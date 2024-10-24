import os
from typing import List, Union

from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from store import BaseVectorStore


class FAISSStore(BaseVectorStore):
    def __init__(self, embeddings: Embeddings, index: FAISS = None):
        """
        初始化 FAISSStore。

        :param embeddings: 用于生成向量的嵌入模型
        :param index: 可选，已有的 FAISS 索引
        """
        self.index: Union[FAISS, None] = index
        self.embeddings: Embeddings = embeddings

    def load(self, file_path: str) -> None:
        """从指定文件路径加载 FAISS 索引。"""
        print(f"从文件 {file_path} 加载 FAISS 索引...")
        self.index = FAISS.load_local(file_path, self.embeddings, allow_dangerous_deserialization=True)
        print("FAISS 索引加载完成。")

    def save(self, file_path: str) -> None:
        """将当前的 FAISS 索引保存到指定文件路径。"""
        if self.index is not None:
            print(f"将 FAISS 索引保存到文件 {file_path}...")
            self.index.save_local(file_path)
            print("FAISS 索引保存完成。")
        else:
            raise ValueError("FAISS 索引为空，无法保存。")

    def as_retriever(self) -> VectorStoreRetriever:
        """将当前的向量存储转换为检索器。"""
        if self.index is not None:
            return self.index.as_retriever()
        else:
            raise ValueError("FAISS 索引未初始化，无法转换为检索器。")

    def add_documents(self, documents: List[Document], batch_size: int = 10) -> None:
        """
        将文档分批添加到 FAISS 索引。

        :param documents: 待添加的文档列表
        :param batch_size: 每个批次处理的文档数量
        """
        total_docs = len(documents)
        print(f"开始添加 {total_docs} 条文档到 FAISS 索引。")

        if self.index is None:
            # 使用第一个批次创建索引
            initial_batch = documents[:batch_size]
            print(f"索引不存在，使用前 {len(initial_batch)} 条文档创建索引...")
            self.index = FAISS.from_documents(initial_batch, self.embeddings)
            print("索引创建完成。")
            start_idx = batch_size
        else:
            print("索引已存在，继续添加文档...")
            start_idx = 0

        # 分批添加剩余的文档
        for i in range(start_idx, total_docs, batch_size):
            batch_docs = documents[i:i + batch_size]
            print(f"正在添加第 {i + 1} 到第 {i + len(batch_docs)} 条文档...")
            self.index.add_documents(batch_docs)
        print("所有文档已添加到索引中。")