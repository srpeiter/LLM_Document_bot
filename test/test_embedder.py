from pathlib import Path
from re import search

import lorem as lm
import pytest
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfFileReader, PdfReader

from LLM_UI.custom_types import EmbedderInputTypes, VSInputTypes
from LLM_UI.embedder import VectorDB, _embedder_map, _vector_store_map


@pytest.fixture
def generate_text_chunks():
    # ui_vm = UI_VM()
    curr_dir = Path(__file__).parents[1]
    data_path = curr_dir / "data" / "2pager_thrive.pdf"

    text = ""

    # with open(data_path, "rb") as f:
    #     pdf = PdfFileReader(f)
    #     text += pdf.getPage(0).extract_text()

    pdf = PdfReader(data_path)
    for page in pdf.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.split_text(text)


@pytest.mark.dependency(name="a")
def test_vector_db_init():
    text_chunks = [lm.paragraph() for i in range(5)]

    embedder_name = EmbedderInputTypes.hugging_face
    vs_name = VSInputTypes.chroma_db

    vector_db = VectorDB(
        text_chunks=text_chunks,
        embedder_type=embedder_name,
        vs_type=vs_name,
    )

    assert isinstance(vector_db.get_db, _vector_store_map[vs_name]) and isinstance(
        vector_db.get_embedder(), _embedder_map[embedder_name]
    )


@pytest.mark.dependency(name="b", depends=["a"])
def test_get_sim_vector(generate_text_chunks):
    text_chunks = generate_text_chunks

    vector_db = VectorDB(
        text_chunks=text_chunks,
        embedder_type=EmbedderInputTypes.open_ai,
        vs_type=VSInputTypes.faiss,
    )

    search_result = vector_db.get_sim_vector("Who are the stakeholders of this project")

    assert len(search_result) > 0
