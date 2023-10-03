from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader

from LLM_UI.chain_factory import chain_factory
from LLM_UI.custom_types import EmbedderInputTypes, LLMName, VSInputTypes
from LLM_UI.embedder import VectorDB


class UI_VM(object):
    def __init__(
        self,
        llm_name: LLMName,
        embedder_name: EmbedderInputTypes,
        vector_store_name: VSInputTypes,
        **kwargs,
    ):
        self.params_embedder: Dict[str, Any] = kwargs.get("params_embedder", {})
        self.params_vector_store: Dict[str, Any] = kwargs.get("params_vector_store", {})
        self.params_llm: Dict[str, Any] = kwargs.get("params_llm", {})
        self.params_text_splitter: Dict[str, Any] = kwargs.get(
            "params_text_splitter", {}
        )

        self.llm_name = llm_name
        self.embedder_name = embedder_name
        self.vector_store_name = vector_store_name

        self.vector_db: VectorDB
        self.llm_conv_chain: BaseConversationalRetrievalChain

        if self.params_text_splitter is not None:
            self.text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=self.params_text_splitter.get("chunk_size", 1000),
                chunk_overlap=self.params_text_splitter.get("chunk_overlap", 200),
                length_function=len,
                is_separator_regex=False,
            )

        load_dotenv()

    def process(self, files: List[str]):
        raw_text = self._read_PDF(files)

        text_chunks = self._get_text_chunks(raw_text)

        self.vector_db = self._init_text_db(
            text_chunks,
            self.embedder_name,
            self.vector_store_name,
            params_embedder=self.params_embedder,
            params_vector_store=self.params_vector_store,
        )

        self.llm_conv_chain = chain_factory(
            self.llm_name, self.vector_db, **self.params_llm
        )

    def _read_PDF(self, files: List[str]):
        text = ""
        for file in files:
            pdf = PdfReader(file)
            for page in pdf.pages:
                text += page.extract_text()
        return text

    def _get_text_chunks(self, text: str):
        return self.text_splitter.split_text(text)

    def _init_text_db(
        self,
        text_chunks: List[str],
        embedder_type: EmbedderInputTypes,
        vs_type: VSInputTypes,
        **kwargs,
    ):
        return VectorDB(
            text_chunks=text_chunks,
            embedder_type=embedder_type,
            vs_type=vs_type,
            **kwargs,
        )

    def handle_user_input(self, user_question: str):
        return self.llm_conv_chain({"question": user_question})
