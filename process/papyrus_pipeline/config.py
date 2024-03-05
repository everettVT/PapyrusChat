import yaml


class PapyrusConfig:
    def __init__(self, file_path: str) -> None:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        self.milvus_endpoint = config['milvus']['host']
        self.port = config['milvus']['port']
        self.connection_args = {"host": self.milvus_endpoint, "port": f'{self.port}'}
        self.milvus_address=f'{self.milvus_endpoint}:{self.port}'

        self.openai_key = config['openai']['key']

        self.video_file_dir = config['video_file_dir']
        self.output_dir = config['output_dir']
        self.processed_video_dir_path = config['processed_video_dir_path']
