import argparse
import os
import re
import sys
import typing
import time 

from Katna.image_filters.text_detector import TextDetector
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import Katna.helper_functions as helper

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

from pymilvus import connections

import yaml

# td = TextDetector()
# td.download()

from PIL import Image

import pytesseract

from papyrus_pipeline.chunk_and_persist import index
from papyrus_pipeline import PapyrusConfig


def extract(video_file_path: str, num_frames: int, output_path: str):
    vd = Video()
    # initialize diskwriter to save data at desired location
    diskwriter = KeyFrameDiskWriter(location=output_path)
    print(f"Input video file path = {video_file_path}")
    vd.extract_video_keyframes(
        no_of_frames=num_frames, file_path=video_file_path,
        writer=diskwriter
    )


def ocr(img_paths: typing.List[str], config: PapyrusConfig):
    # use a local lib for some initial quality validation before sending to openai

    image_documents = SimpleDirectoryReader(img_paths).load_data()
    responses = []

    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4-vision-preview", api_key=config.openai_key, max_new_tokens=1500
    )
    for image_doc in image_documents:
        # TODO: handle exception and retry
        response_1 = openai_mm_llm.complete(
            prompt="Perform OCR on the pages shown. Only provide text from the images in the response. Give the response as json, with the text under a key called output",
            image_documents=[image_doc],
        )
        # print(response_1)
        json_txt = response_1.text.replace('```', '').replace("\'", '')
        json_txt = json_txt.replace('\\n', ' ').replace('\\\\n', ' ')
        json_txt = re.findall(r'.*output["\']\s*:\s*([^}]+)', json_txt)
        print(json_txt)
        if len(json_txt) > 0 and len(json_txt[0].strip()) > 5:
            responses.append(json_txt[0])
    return responses


def process(video_file_path: str, output_dir: str, config: PapyrusConfig) -> str:
    # assume like 0.70 flips per second
    vid_info = helper.get_video_info(video_file_path)
    num_frames = int(vid_info[1] * 0.70)
    extract(video_file_path, num_frames, output_dir)
    ocr_op = ocr(output_dir, config)
    print(ocr_op)

    txt_dir = f"{output_dir}/text/"
    os.makedirs(txt_dir, exist_ok=True)
    for i, s in enumerate(ocr_op):
        with open(os.path.join(txt_dir, f"{i}.txt"), mode='wt') as f:
            f.write(s)
    return txt_dir


def main():
    # TODO: take metadtaa files as input, containign author, file name, book name, etc
    config = PapyrusConfig("config.yaml")
    vid_dir = config.video_file_dir
    output_dir = config.output_dir
    processed_video_dir_path = config.processed_video_dir_path
    # TODO: get from metadata
    author_name: str = "author"
    for p in [vid_dir, output_dir, processed_video_dir_path]:
        os.makedirs(p, exist_ok=True)

    while True:
        # look for any files in the directory
        for vid_name in os.listdir(vid_dir):
            video_file_path = os.path.join(vid_dir, vid_name)
            temp_output_dir = os.path.join(output_dir, str(int(time.time())))
            txt_dir = process(video_file_path, temp_output_dir, config)
            # move to the other folder when done
            os.rename(video_file_path, os.path.join(processed_video_dir_path, vid_name))
            # TODO: take that metdata, create a per-author index
            index(collection_name=author_name, doc_path=txt_dir, config=config, metadata={})
            # TODO publish it's done
        time.sleep(10)


if __name__ == "__main__":
    main()
