import re
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv
import os
import openai 
import gradio as gr
from numpy.linalg import norm
from typing import List
from llama_index.core import Settings
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import (SentenceSplitter, SemanticSplitterNodeParser)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode, TransformComponent
# from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
import chromadb

_ = load_dotenv(find_dotenv())


OPENAI_MODEL = os.getenv('DEFAULT_MODEL_NAME')

EMBED_DIMENSION=512
Settings.llm = OpenAI(model=OPENAI_MODEL, temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=EMBED_DIMENSION, api_base="https://api.vansai.cn/v1/" )

path = "./OnlyOne/"
reader = SimpleDirectoryReader(input_dir=path, required_exts=['.pdf'])
documents = reader.load_data()
print(documents[0])

# 创建一个 Chroma 向量存储
chroma_client = chromadb.PersistentClient(path="./data/chroma_db") 
# 判断是否存在名为 "mybook" 的集合，如果存在则删除，然后创建新的集合
if "mybook" in chroma_client.list_collections():
    chroma_client.delete_collection("mybook")
chroma_collection = chroma_client.create_collection("mybook")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

class TextCleaner(TransformComponent):
    """
    用于数据摄取管道中的转换。
    清理文本中的杂乱内容。
    """
    def __call__(self, nodes, **kwargs) -> List[BaseNode]:
        for node in nodes:
            node.text = node.text.replace('\t', ' ')  # 将制表符替换为空格
            node.text = node.text.replace(' \n', ' ')  # 将段落分隔符替换为空格
        return nodes
    
# 创建基础分割器
base_splitter = SentenceSplitter(chunk_size=512)

# 创建语义分割器，并设置 base_splitter
semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=Settings.embed_model,
    base_splitter=base_splitter,  # 添加基础分割器
    include_metadata=True,
    include_prev_next_rel=True
)

# 数据摄取管道实例：
# 节点解析器，自定义转换器，向量存储和文档
pipeline = IngestionPipeline(
    transformations=[
        semantic_splitter,
        TextCleaner()
    ],
    vector_store=vector_store,
    documents=documents
)

# 运行管道以获取节点
nodes = pipeline.run()

# BM25 检索器
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=2
)

# 向量检索器
index = VectorStoreIndex(nodes)
vector_retriever = index.as_retriever(similarity_top_k=2)

# 融合两个检索
retriever = QueryFusionRetriever(
    retrievers=[
        vector_retriever,  # 向量检索器
        bm25_retriever  # BM25检索器
    ],
    retriever_weights=[
        0.6,  # 向量检索器的权重
        0.4  # BM25检索器的权重
    ],
    num_queries=1,  # 查询次数
    mode='dist_based_score',  # 使用基于距离的评分模式
    use_async=False  # 是否使用异步
)


from openai import OpenAI
# OpenAI API密钥
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.base_url="https://api.vansai.cn/v1/" 


client = OpenAI()
MODEL = os.getenv('DEFAULT_MODEL_NAME')
#     # 使用 LLM 回答问题的方法
def get_completion(prompt, model=MODEL):
    '''封装 openai 接口'''
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
        stream=True
    )
    return response
def build_prompt(prompt_template, **kwargs):
    """将 Prompt 模板赋值"""
    inputs = {}
    for k, v in kwargs.items():
        if isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n\n'.join(v)
        else:
            val = v
        inputs[k] = val
    return prompt_template.format(**inputs)

prompt_template = """
根据下述给定的已知信息回答用户问题。

已知信息:
{context}

用户问：
{query}


你是李清照，一个infj人格类型的寓居南京的诗人。请以诗意的语言来回答问题，但答案不要含有“诗意”这两个字。

如果已知信息不包含用户问题的答案，或者已知信息不足以回答用户的问题，请委婉拒绝回答。
如果问题与诗歌无关，请委婉拒绝回答。

注意区分已知信息中诗的题目，诗的内容和诗的作者。
已知信息中用《》符号括起来的是诗的题目，诗的内容在下面，诗的作者在后面。



"""


history=" "

def process(user_query):
    global history
    history=history+user_query
    # 1. 基于向量的相似度检索
    #search_results = search_similar(history, 20)

    search_results = retriever.retrieve(history)[0].node.text
    # 2. 构建 Prompt
    prompt = build_prompt(prompt_template, context=search_results, query=user_query)
    # 3. 调用 LL
    #response=prompt
    response = get_completion(prompt)
    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message += chunk.choices[0].delta.content
            yield partial_message
    history = history + partial_message
    return response

demo = gr.Interface(
    fn=process, 
    inputs="textbox", 
    outputs="textbox",
    live=False
    )

demo.launch(share=True)