import yaml


class Config:
    def __init__(self, file_path: str) -> None:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        self.milvus_endpoint = config['milvus']['host']
        self.port = config['milvus']['port']
        self.connection_args = {"host": self.milvus_endpoint, "port": f'{self.port}'}
        self.milvus_address=f'{self.milvus_endpoint}:{self.port}'

        self.openai_key = config['openai']['key']
