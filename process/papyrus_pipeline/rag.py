from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.indices import MultiModalVectorStoreIndex


milvus_client = connections.connect("default", host=config['milvus']['host'], port=config['milvus']['port'])
client = Foo 
meta_store = MilvusVectorStore(client= milvus_client, collection_name = "meta_collection" )
image_store = MilvusVectorStore(client= milvus_client, collection_name = "images_collection")


MilvusVectorStore(dim=1536, collection_name="lyft", overwrite=False)

p = f"Analyze the following images and transcribe any text that is found. Format your response as JSON. "

openai_mm_llm = OpenAIMultiModal(
        model="gpt-4-vision-preview",
        api_key=config['openai']['key'],
        max_new_tokens=500,
        temperature=0.3,
    )    

response_gpt4v = openai_mm_llm.complete(
        prompt=p,
        image_documents=img_docs,
    )

def sink(config, img_docs):
    

def query(query):
    

print(response_gpt4v)