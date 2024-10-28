import os
from typing import List, Union, Any

from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from store import BaseVectorStore


class FAISSStore(BaseVectorStore):
    def __init__(self, embedding_model_name: str, index: FAISS = None):
        """
        初始化 FAISSStore。

        :param embedding_model_name: 用于生成向量的嵌入模型名称
        :param index: 可选，已有的 FAISS 索引
        """
        self.embedding_model_name = embedding_model_name
        self.index: Union[FAISS, None] = index
        self._embeddings: Union[Embeddings, None] = None  # 初始化 _embeddings 属性

    @property
    def __embeddings(self):
        if self._embeddings is None:
            print("初始化嵌入模型...")
            self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        return self._embeddings

    def load(self, file_path: str) -> None:
        """从指定文件路径加载 FAISS 索引。"""
        print(f"从文件 {file_path} 加载 FAISS 索引...")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在，无法加载索引。")
        self.index = FAISS.load_local(file_path, self.__embeddings, allow_dangerous_deserialization=True)
        print("FAISS 索引加载完成。")

    def save(self, file_path: str) -> None:
        """将当前的 FAISS 索引保存到指定文件路径。"""
        if self.index is not None:
            print(f"将 FAISS 索引保存到文件 {file_path}...")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self.index.save_local(file_path)
            print("FAISS 索引保存完成。")
        else:
            raise ValueError("FAISS 索引为空，无法保存。")

    def as_retriever(self, k: int = 4) -> VectorStoreRetriever:
        """将当前的向量存储转换为检索器。"""
        if self.index is not None:
            return self.index.as_retriever(search_kwargs={'k': k})
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

        if total_docs == 0:
            print("没有文档可添加。")
            return

        if self.index is None:
            print("索引不存在，正在创建新的索引...")
            initial_batch = documents[:batch_size]
            self.index = FAISS.from_documents(initial_batch, self.__embeddings)
            print("索引创建完成。")
            start_idx = len(initial_batch)
        else:
            print("索引已存在，继续添加文档...")
            start_idx = 0

        # 分批添加剩余的文档
        for i in range(start_idx, total_docs, batch_size):
            batch_docs = documents[i:i + batch_size]
            print(f"正在添加第 {i + 1} 到第 {i + len(batch_docs)} 条文档...")
            self.index.add_documents(batch_docs)
        print("所有文档已添加到索引中。")
