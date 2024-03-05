import json
import os
import time
import typing
from collections import defaultdict

from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from pymilvus import connections, utility
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain.chains import RetrievalQA

from milvus import MilvusServer, MilvusServerConfig

import yaml

from sklearn.feature_extraction.text import TfidfVectorizer

from papyrus_pipeline import PapyrusConfig


def chunk_and_persist(collection_name: str, docs_list: typing.List[Document], connection_args: typing.Mapping[str, str], embeddings):
    return Milvus.from_documents(
        documents=docs_list,
        embedding=embeddings,
        # drop_old=True,
        connection_args=connection_args,
        collection_name=collection_name)


def drop_collection(collection_name: str, config: PapyrusConfig):
    connections.connect(alias="del", address=config.milvus_address)
    # connections.connect("del",
    #                 uri=milvus_endpoint,
    #                 token=zilliz_api_key)
    utility.drop_collection(collection_name, using="del")


def chunk_text(docs: typing.List[Document], chunk_size: int, chunk_overlap: int):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)


def load_db(collection_name: str, embeddings, connection_args: typing.Mapping[str, str]):
    vector_db = Milvus(
        embeddings,
        connection_args=connection_args,
        collection_name=collection_name,
    )
    return vector_db



def index_docs_at_path(collection_name: str, dir_path: str, embeddings):
    loader = DirectoryLoader(dir_path, glob="**/*.txt")
    docs = loader.load()
    index_docs(collection_name, docs, embeddings)


def index_docs(collection_name: str, docs: typing.List[Document], embeddings, config: PapyrusConfig):
    chunked_docs = chunk_text(docs, chunk_size=512, chunk_overlap=0)
    chunk_and_persist(collection_name, chunked_docs, connection_args=config.connection_args, embeddings=embeddings)


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


def index(collection_name: str, doc_path: str, config: PapyrusConfig):
    import sys
    # drop_collection(collection_name)
    embeddings = OpenAIEmbeddings(openai_api_key=config.openai_key)
    file_names = os.listdir(doc_path)
    docs = []
    for f_name in file_names:
        with open(os.path.join(doc_path, f_name), mode='rt', encoding='utf8') as f:
            print(f"Reading {f_name}")
            doc_contents = f.read()
            docs.append(Document(page_content=doc_contents))
    index_docs(collection_name, docs, embeddings, config)
    print("Indexed")
