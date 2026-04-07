---
name: rag-knowledge-base
description: "RAG 知识库构建助手。向量数据库、Embedding、检索增强生成。触发词：rag、向量、embedding、知识库。"
metadata: {"openclaw": {"emoji": "🧠"}}
---

# RAG Knowledge Base Skill

## 功能说明

构建完整的 RAG（检索增强生成）知识库系统。

## 架构概览

```
用户查询 → Embedding模型 → 向量数据库检索 → 上下文组装 → LLM生成 → 回答
```

## 核心技术栈

### Embedding 模型选择

| 模型 | 维度 | 特点 | 适用场景 |
|------|------|------|----------|
| text-embedding-ada-002 | 1536 | OpenAI官方，稳定 | 通用场景 |
| text-embedding-3-small | 1536/256 | 新版，性价比高 | 快速场景 |
| text-embedding-3-large | 3072 | 高精度 | 高质量场景 |
| BGE-large-zh | 1024 | 中文优化 | 中文知识库 |
| M3E | 768 | 开源免费 | 自托管 |

## 完整实现

### 1. 文档切分

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(texts, chunk_size=500, chunk_overlap=50):
    """智能文档切分"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        length_function=len
    )
    return splitter.create_documents(texts)

# 高级：按语义切分
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

chunker = SemanticChunker(
    embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh"),
    breakpoint_threshold_type="percentile"
)
chunks = chunker.create_documents([long_text])
```

### 2. 向量存储

```python
# OpenAI + Chroma（轻量）
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader

# 加载文档
loader = TextLoader("document.txt")
documents = loader.load()

# 切分
texts = split_documents([d.page_content for d in documents])

# 创建向量库
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)
vectorstore.persist()

# 检索
results = vectorstore.similarity_search("查询内容", k=5)
```

```python
# 开源自托管（Milvus + BGE）
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-zh')
embeddings = model.encode(["文本"], normalize_embeddings=True)

# 连接到Milvus
connections.connect(host="localhost", port="19530")
collection = Collection("knowledge_base")
collection.load()

# 搜索
search_params = {"metric_type": "COSINE", "params": {"ef": 128}}
results = collection.search(
    data=[embeddings[0].tolist()],
    anns_field="embedding",
    param=search_params,
    limit=5,
    output_fields=["content", "source"]
)
```

### 3. 检索策略

```python
# 混合检索（关键词 + 向量）
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25 关键词检索
bm25_retriever = BM25Retriever.from_texts(chunks)
bm25_retriever.k = 5

# 向量检索
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 融合（RRF算法）
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]  # 权重配比
)

results = ensemble_retriever.get_relevant_documents("查询")
```

```python
# 重排序（Rerank）
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")

def rerank(query, documents, top_k=3):
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

# 使用
initial_results = ensemble_retriever.get_relevant_documents(query)
final_results = rerank(query, initial_results)
```

### 4. RAG 应用

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 自定义提示词
template = """基于以下上下文回答问题。如果上下文中没有相关信息，请如实说明。

上下文：
{context}

问题：{question}

回答："""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# 构建QA链
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=ensemble_retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

result = qa_chain({"query": "用户问题"})
print(result["result"])
```

### 5. 高级：Query 改写

```python
# HyDE（Hypothetical Document Embeddings）
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

hyde_prompt = PromptTemplate(
    template="假设你是一个专家，请根据问题写一段可能的回答：\n\n问题：{question}\n\n回答：",
    input_variables=["question"]
)
hyde_chain = LLMChain(llm=llm, prompt=hyde_prompt)

# 用假想答案检索
hypothetical_doc = hyde_chain.run(question)
results = vectorstore.similarity_search(hypothetical_doc, k=5)
```

## 性能优化

### 索引优化
```python
# Chroma 索引优化
vectorstore = Chroma(
    client=chromadb.PersistentClient(path="./db"),
    embedding_function=OpenAIEmbeddings(),
    collection_metadata={"hnsw:space": "cosine", "hnsw:ef_construction": 200, "hnsw:ef": 128}
)
```

### 缓存优化
```python
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())
```

## 评估指标

| 指标 | 含义 | 工具 |
|------|------|------|
| Retriever Recall | 召回率 | RAGAS |
| Retriever Precision | 精确率 | RAGAS |
| Answer Faithfulness | 答案忠实度 | RAGAS |
| Answer Relevancy | 答案相关性 | RAGAS |
| Context Precision | 上下文精确率 | RAGAS |

## 常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 检索不到相关内容 | chunk_size过大 | 减小到200-500 |
| 答案不准确 | Embedding模型不匹配 | 换用中文模型 |
| 检索太慢 | 索引未优化 | 调参hnsw:ef |
| 上下文截断 | token超限 | 增加window或summary |
