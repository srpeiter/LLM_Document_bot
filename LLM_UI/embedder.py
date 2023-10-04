from typing import Any, Dict, List

from langchain.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
    OpenAIEmbeddings,
)
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import FAISS, Chroma

from LLM_UI.custom_types import EmbedderInputTypes, VSInputTypes, VSTypes

_vector_store_map: Dict[str, Any] = {"faiss": FAISS, "chroma_db": Chroma}

_embedder_map: Dict[str, Any] = {
    "hugging_face": HuggingFaceEmbeddings,
    "hugging_face_instruct": HuggingFaceInstructEmbeddings,
    "open_ai": OpenAIEmbeddings,
}


class VectorDB:
    def __init__(
        self,
        text_chunks: List[str],
        embedder_type: EmbedderInputTypes,
        vs_type: VSInputTypes,
        **kwargs,
    ):
        self.params_embedder: Dict[str, Any] = kwargs.get("params_embedder", {})
        self.params_vector_store: Dict[str, Any] = kwargs.get("params_vector_store", {})
        self._embedder: Embeddings
        self._vector_store: VSTypes
        self.init_embedder(embedder_type, **self.params_embedder)
        self.vector_store = vs_type
        self._db = self.vector_store.from_texts(
            text_chunks, embedding=self.get_embedder(), **self.params_vector_store
        )

    def get_embedder(self) -> Embeddings:
        return self._embedder

    def init_embedder(self, type: EmbedderInputTypes, **kwargs):
        self._embedder = _embedder_map[type](**kwargs)

    @property
    def vector_store(self) -> VSTypes:
        return self._vector_store

    @vector_store.setter
    def vector_store(self, type: VSInputTypes):
        self._vector_store = _vector_store_map[type]

    @property
    def get_db(self):
        return self._db

    def get_sim_vector(self, query: str):
        return self._db.similarity_search(
            query=query,
        )
