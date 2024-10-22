from typing import List, Dict, Union, Any
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from store import BaseVectorStore


class FAISSStore(BaseVectorStore):
    def __init__(self, embeddings: Embeddings, index: FAISS = None):
        self.index: Union[FAISS, None] = index
        self.embeddings: Embeddings = embeddings
        self.index_to_docstore_id: Dict[int, str] = {}

    def load(self, file_path: str) -> None:
        """从文件路径加载FAISS索引"""
        print(f"从文件 {file_path} 加载 FAISS 索引")
        self.index = FAISS.load_local(file_path, self.embeddings, allow_dangerous_deserialization=True)

    def save(self, file_path: str) -> None:
        """将FAISS索引保存到文件路径"""
        if self.index is not None:
            print(f"将 FAISS 索引保存到文件 {file_path}")
            self.index.save_local(file_path)
        else:
            raise ValueError("FAISS 索引为空，无法保存")

    def add_documents(self, documents: List[Document], batch_size: int = 10) -> None:
        """将文档分批添加到 FAISS 索引"""
        print("从文档添加到 FAISS 索引")
        # 如果索引不存在，先创建索引
        if self.index is None:
            # 创建索引时，分批处理文档，防止内存溢出
            print(f"索引不存在，分批创建索引，每批次 {batch_size} 条文档，总共 {len(documents)} 条文档")
            for i in range(0, len(documents), batch_size):
                print(f"正在处理第 {i} 到 {i + batch_size} 条文档")
                batch_docs = documents[i:i + batch_size]
                if self.index is None:
                    self.index = FAISS.from_documents(batch_docs, self.embeddings)
                else:
                    self.index.add_documents(batch_docs)
        else:
            # 如果索引已经存在，分批添加文档
            print(f"索引已存在，分批添加文档，每批次 {batch_size} 条文档")
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                self.index.add_documents(batch_docs)

    def search(self, query: str, top_k: int = 5) -> List[Any]:
        """进行相似性搜索，返回前K个结果"""
        if self.index is None:
            raise ValueError("FAISS 索引未加载")

        print(f"进行相似性搜索，查询：{query}")
        results = self.index.similarity_search_with_score(query, k=top_k)
        return results
