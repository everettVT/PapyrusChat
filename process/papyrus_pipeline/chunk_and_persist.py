import json
import os
import time
import typing
from collections import defaultdict

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document

from langchain.chains.query_constructor.base import AttributeInfo
from pymilvus import connections, utility

from sklearn.feature_extraction.text import TfidfVectorizer

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext
import openai

from papyrus_pipeline import PapyrusConfig


def index_docs(docs_list: typing.List[Document], embed_model: BaseEmbedding, chunk_size: int, storage_context: StorageContext):
    node_parser = SentenceSplitter(chunk_size=chunk_size)
    nodes = node_parser.get_nodes_from_documents(docs_list)
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

    return VectorStoreIndex.from_documents(
        documents=docs_list,
        storage_context=storage_context)


def drop_collection(collection_name: str, config: PapyrusConfig):
    connections.connect(alias="del", address=config.milvus_address)
    utility.drop_collection(collection_name, using="del")


def score2_sub(a, b):
    word_a = a.split()
    word_b = set(b.split())

    d = defaultdict(int)
    for word in word_a:
        if word in word_b:
            d[word] += 1

    return sum(d.values()) / len(word_a)


def score2_m(a, b):
    return max(score2_sub(a,b), score2_sub(b,a))


def score2(docs):
    scores = [None] * len(docs)
    for i in range(0, len(docs)):
        if scores[i] is None:
            scores[i] = []
        for j in range(0, len(docs)):
            scores[i].append(score2_m(docs[i], docs[j]))
    return scores


def index(collection_name: str, doc_path: str, metadata: typing.Mapping[str, str], config: PapyrusConfig):
    openai.api_key = config.openai_key
    vector_store = MilvusVectorStore(dim=1536, overwrite=False, uri=config.milvus_address, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OpenAIEmbedding(api_key=config.openai_key)
    file_names = os.listdir(doc_path)
    docs = []
    for f_name in file_names:
        with open(os.path.join(doc_path, f_name), mode='rt', encoding='utf8') as f:
            print(f"Reading {f_name}")
            doc_contents = f.read()
            doc = Document(text=doc_contents, metadata=metadata)
            docs.append(doc)
    index_docs(docs, embed_model, chunk_size=256, storage_context=storage_context)
    print("Indexed")
