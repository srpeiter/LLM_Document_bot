from pathlib import Path
from typing import List

from IPython import embed
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader

from LLM_UI.custom_types import EmbedderInputTypes, VSInputTypes
from LLM_UI.embedder import VectorDB


class UI_VM(object):
    def __init__(self):
        self.vector_db: VectorDB

        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

    def import_PDF(self, file: str):
        raw_text = self._read_PDF(file)

        text_chunks = self._get_text_chunks(raw_text)

        self._load_text_db(text_chunks=text_chunks)

    def _read_PDF(self, file: str):
        pdf = PdfReader(file)

        text: str = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text

    def _get_text_chunks(self, text: str):
        return self.text_splitter.split_text(text)

    def _load_text_db(
        self,
        text_chunks: List[str],
        embedder_type: EmbedderInputTypes,
        vs_type: VSInputTypes,
    ):
        self.vector_db = VectorDB(
            text_chunks=text_chunks, embedder_type=embedder_type, vs_type=vs_type
        )

    def pass_user_input():
        pass
