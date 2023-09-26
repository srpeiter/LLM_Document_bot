from typing import Any, Dict, List

from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import FAISS, Chroma

from LLM_UI.custom_types import EmbedderInputTypes, EmbedderTypes, VSInputTypes, VSTypes

_vector_store_map: Dict[str, Any] = {"faiss": FAISS, "chroma_db": Chroma}

_embedder_map: Dict[str, Any] = {
    "hugging_face": HuggingFaceEmbeddings,
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
        self._embedder: Embeddings
        self._vector_store: VSTypes
        self.embedder = embedder_type
        self.vector_store = vs_type
        self.db = self.vector_store.from_texts(text_chunks, embedding=self.embedder)

    @property
    def embedder(self) -> Embeddings:
        return self._embedder

    @embedder.setter
    def embedder(self, type: EmbedderInputTypes):
        self._embedder = _embedder_map[type]()

    @property
    def vector_store(self) -> VSTypes:
        return self._vector_store

    @vector_store.setter
    def vector_store(self, type: VSInputTypes):
        self._vector_store = _vector_store_map[type]

    def get_sim_vector(self, query: str):
        return self.db.similarity_search(query=query)
