import lorem as lm
import pytest
from langchain.chains import ConversationalRetrievalChain

from LLM_UI.chain_factory import chain_factory
from LLM_UI.custom_types import EmbedderInputTypes, LLMName, VSInputTypes
from LLM_UI.embedder import VectorDB


@pytest.fixture
def init_vector_db():
    text_chunks = [lm.paragraph() for i in range(5)]

    embedder_name = EmbedderInputTypes.open_ai
    vs_name = VSInputTypes.faiss

    return VectorDB(
        text_chunks=text_chunks,
        embedder_type=embedder_name,
        vs_type=vs_name,
    )


def test_chain_factory(init_vector_db: VectorDB):
    vector_db = init_vector_db
    llm_name = LLMName.chat_open_ai

    llm_chain = chain_factory(llm_name, vector_db)

    assert isinstance(llm_chain, ConversationalRetrievalChain)
