from enum import Enum
from typing import Type, Union

from langchain import FAISS
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma


class EmbedderInputTypes(str, Enum):
    hugging_face = "hugging_face"
    open_ai = "open_ai"


class VSInputTypes(str, Enum):
    faiss = "faiss"
    chroma_db = "chroma_db"


VSTypes = Union[FAISS, Chroma]
EmbedderTypes = Union[HuggingFaceEmbeddings, OpenAIEmbeddings]
