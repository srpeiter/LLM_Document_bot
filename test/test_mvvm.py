from pathlib import Path
from typing import List

import lorem as lm
import pytest

from LLM_UI.custom_types import EmbedderInputTypes, LLMName, VSInputTypes
from LLM_UI.embedder import _embedder_map, _vector_store_map
from LLM_UI.mvvm import UI_VM

# ui_vm = UI_VM()
curr_dir = Path(__file__).parents[1]
data_path = curr_dir / "data" / "llm_paper.pdf"


@pytest.fixture
def _init_ui_vm():
    embedder_name = EmbedderInputTypes.hugging_face
    vs_name = VSInputTypes.faiss
    llm_name = LLMName.chat_open_ai

    return UI_VM(llm_name, embedder_name, vs_name)


@pytest.mark.dependency(name="a")
def test_read_pdf(_init_ui_vm: UI_VM):
    ui_vm = _init_ui_vm
    text: str = ui_vm._read_PDF(file=str(data_path))

    assert text != None and type(text) == str, print("error")


@pytest.mark.dependency(name="b", depends=["a"])
def test_get_text_chunks(_init_ui_vm: UI_VM):
    ui_vm = _init_ui_vm
    text: str = lm.text()

    chunks: List[str] = ui_vm._get_text_chunks(text)

    assert chunks != None and type(chunks) == list


@pytest.mark.dependency(name="c", depends=["b"])
def test_init_text_db(_init_ui_vm: UI_VM):
    ui_vm = _init_ui_vm
    text_chunks: List[str] = [lm.paragraph() for _ in range(5)]

    embedder_name = EmbedderInputTypes.open_ai
    vs_name = VSInputTypes.faiss
    vector_db = ui_vm._init_text_db(text_chunks, embedder_name, vs_name)

    assert isinstance(vector_db.get_db, _vector_store_map[vs_name]) and isinstance(
        vector_db.embedder, _embedder_map[embedder_name]
    )


@pytest.fixture
def embed_text(_init_ui_vm: UI_VM):
    ui_vm = _init_ui_vm

    ui_vm.process(str(data_path))

    return ui_vm


def test_handle_user_input(embed_text: UI_VM):
    ui_vm = embed_text

    user_question = "Give me a summary of the text"
    response = ui_vm.handle_user_input(user_question)

    assert isinstance(response["answer"], str) and len(response["answer"]) > 0
