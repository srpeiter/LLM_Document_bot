from pathlib import Path
from typing import List

import lorem as lm
import pytest
from numpy import vstack

from LLM_UI.embedder import _embedder_map, _vector_store_map
from LLM_UI.mvvm import UI_VM

ui_vm = UI_VM()
curr_dir = Path(__file__).parents[1]
data_path = curr_dir / "data" / "llm_paper.pdf"


@pytest.mark.dependency(name="a")
def test_read_pdf():
    ui_vm = UI_VM()
    text: str = ui_vm._read_PDF(data_path)

    assert text != None and type(text) == str, print("error")


@pytest.mark.dependency(name="b", depends=["a"])
def test_get_text_chunks():
    text: str = lm.text()

    chunks: List[str] = ui_vm._get_text_chunks(text)

    assert chunks != None and type(chunks) == list


@pytest.mark.dependency(name="c", depends=["b"])
def test_load_text_db():
    text_chunks: List[str] = [lm.paragraph() for _ in range(5)]

    embedder_type = "hugging_face"
    vs_type = "chroma_db"
    ui_vm._load_text_db(text_chunks, embedder_type, vs_type)

    assert isinstance(ui_vm.vector_db.db, _vector_store_map[vs_type]) and isinstance(
        ui_vm.vector_db.embedder, _embedder_map[embedder_type]
    )
