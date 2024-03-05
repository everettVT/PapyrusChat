from milvus import MilvusServer, MilvusServerConfig

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal





milvus_endpoint = config['milvus']['host']  #'localhost' #'https://in03-edeea0a7bd3937f.api.gcp-us-west1.zillizcloud.com'
port = config['milvus']['port']
openai_key = config['openai']['key']

connection_args = {"host": milvus_endpoint, "port": f'{port}'}
milvus_address=f'{milvus_endpoint}:{port}'