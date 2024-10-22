import os
from langchain_community.document_loaders import JSONLoader
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from store.faiss import FAISSStore

def initialize_faiss_index_from_file(store: FAISSStore, spliter: TextSplitter, data_file_path: str,
                                     index_file_path: str) -> FAISSStore:
    """
   从文件中初始化 FAISS 索引。如果索引文件存在，则从文件加载索引。
   否则，从数据文件创建新索引并将其保存到索引文件。

   参数:
       store (FAISSStore): 用于索引的 FAISS 存储。
       spliter (TextSplitter): 用于分割文档的文本分割器。
       data_file_path (str): 包含文档的数据文件的路径。
       index_file_path (str): 加载或保存索引的索引文件路径。

   返回:
       FAISSStore: 初始化的 FAISS 存储。
   """
    if os.path.exists(index_file_path):
        print("从本地加载 FAISS 索引...")
        store.load(index_file_path)
    else:
        print("本地没有 FAISS 索引，正在创建...")
        loader = JSONLoader(
            file_path=data_file_path,
            jq_schema='.data[]',
            content_key='content',
            metadata_func=lambda record, metadata:
            {**metadata, "source": record.get("url"), "title": record.get("title"), "date": record.get("date")}
        )
        docs = loader.load()
        print("分割文档并添加到索引")
        split_docs = spliter.split_documents(docs)
        store.add_documents(split_docs)
        print("保存 FAISS 索引到本地...")
        store.save(index_file_path)
    return store


if __name__ == '__main__':
    question = "学校“十五五”发展规划的主要目标是什么？"
    llm = OllamaLLM(model="qwen2.5:7b-instruct")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    # 创建 ChatPromptTemplate
    system_prompt = ("你是中国矿业大学的问答助手。"
                     "使用以下从中国矿业大学的网站上检索到的上下文片段来回答问题。"
                     "如果你不知道答案，就说你不知道。"
                     "最多用三个句子，不要说'根据提供的信息'等相关描述，并保持回答简洁。"
                     "\n\n"
                     "{context}"
                     )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    vector_store = initialize_faiss_index_from_file(store=FAISSStore(embeddings),
                                                    spliter=RecursiveCharacterTextSplitter(chunk_size=2000,
                                                                                           chunk_overlap=100),
                                                    data_file_path='data/news.json',
                                                    index_file_path="faiss_index")

    # 进行相似性搜索
    vector_result = vector_store.search(question, top_k=100)

    # 创建检索和文档链
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(vector_store.index.as_retriever(), document_chain)

    # 执行问题查询
    response = retrieval_chain.invoke({"input": question})
    print(f"Answer: {response['answer']}")
    print()
