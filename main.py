import os
import pickle  # 用于序列化
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chain_callback import ChainCallback
from store.bm25 import BM25Store
from store.faiss import FAISSStore
from langchain.retrievers import EnsembleRetriever

load_dotenv()  # 加载 .env 文件
from langchain_openai import ChatOpenAI


def format_docs(doc_list: List[Document]):
    formatted_docs = []
    index = 1
    for doc in doc_list:
        source = doc.metadata.get("source", "未知来源")  # 从 metadata 获取文档的来源
        content_with_citation = f"第{index}篇文章(来源: {source})：\n\n{doc.page_content}"
        formatted_docs.append(content_with_citation)
        index += 1  # 增加索引
    return "\n\n".join(formatted_docs)


def load_all_data(data_path, serialize_document_path):
    # 检查是否存在序列化后的文档
    if os.path.exists(serialize_document_path):
        print("从本地加载序列化的文档...")
        with open(serialize_document_path, 'rb') as f:
            split_docs = pickle.load(f)
    else:
        print("序列化的文档不存在，正在加载和分割原始文档...")
        loader = JSONLoader(
            file_path=data_path,
            jq_schema='.data[]',
            content_key='content',
            metadata_func=lambda record, metadata:
            {**metadata, "source": record.get("url"), "title": record.get("title"), "date": record.get("date")}
        )
        docs = loader.load()
        print("分割文档...")
        spliter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        split_docs = spliter.split_documents(docs)
        print("保存分割后的文档到本地...")
        os.makedirs(os.path.dirname(serialize_document_path), exist_ok=True)
        with open(serialize_document_path, 'wb') as f:
            pickle.dump(split_docs, f)
    return split_docs


if __name__ == '__main__':
    llm = ChatOpenAI(
        openai_api_base="https://ark.cn-beijing.volces.com/api/v3",
        model_name="ep-20241024122147-th778"
    )

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # 创建 ChatPromptTemplate
    system_prompt = """你是中国矿业大学的问答助手。用以下几段检索到的信息回答问题，保持答案简洁。
                     问题：{question}
                     信息：{context}

                     答案："""

    prompt = ChatPromptTemplate.from_template(system_prompt)

    print("加载数据...")
    docs_all = load_all_data(data_path='data/raw/cumt_news.json',
                             serialize_document_path='data/serialized/serialized_docs.pkl')

    print("创建 faiss_retriever...")
    faiss_store = FAISSStore(embeddings)
    faiss_store.load_or_create_index(index_path="data/index/faiss", documents=docs_all)
    faiss_retriever = faiss_store.as_retriever()

    print("创建 bm25_retriever...")
    bm25_store = BM25Store()
    bm25_store.load_or_create_index(index_path='data/index/bm25', documents=docs_all)
    bm25_retriever = bm25_store.as_retriever()

    print("创建 EnsembleRetriever...")
    ensemble_retriever = EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever])

    # 创建 RAG chain
    rag_chain = (
            {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # 执行问题查询
    response = rag_chain.invoke("宋学峰是谁，他最近在干嘛", config={"callbacks": [ChainCallback()]})
    print(response)
