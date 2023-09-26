from abc import ABC, abstractmethod
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from typing import List
from langchain.vectorstores import Chroma

_vector_store_type = {"faiss": FAISS, "chromadb": Chroma}

_embedder_type = {"hugging_face": HuggingFaceEmbeddings, "open_ai": OpenAIEmbeddings}


class EmbedderStore(ABC):
    @abstractmethod
    def __init__(self, text: List[str]):
        self._embedder = None
        self._vector_store = None

    @property
    def embedder(self):
        return self._embedder

    @property
    def vector_store(self):
        return self._vector_store

    @abstractmethod
    def get_sim_vector(self, text_chunks: str):
        ...


class FAISStore(EmbedderStore):
    def __init__(self, text_chunks: List[str]):
        self._embedder = HuggingFaceEmbeddings()
        self._vector_store = FAISS.from_texts(text_chunks, self._embedder)

    def get_vectorstore(text_chunks: str):
        return super().get_vectorstore()
