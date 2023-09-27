from enum import Enum
from typing import Union

from langchain import FAISS
from langchain.vectorstores import Chroma


class EmbedderInputTypes(str, Enum):
    hugging_face = "hugging_face"
    hugging_face_instruct = "hugging_face_instruct"
    open_ai = "open_ai"


class VSInputTypes(str, Enum):
    faiss = "faiss"
    chroma_db = "chroma_db"


class LLMName(str, Enum):
    hugging_face_hub = "hugging_face_hub"
    llama_cpp = "llama_cpp"
    open_ai = "open_ai"
    chat_open_ai = "chat_open_ai"


VSTypes = Union[FAISS, Chroma]
