import lorem as lm

from LLM_UI.embedder import VectorDB


def test_VectorDB_init():
    text_chunks = [lm.paragraph() for i in range(5)]
    vector_db = VectorDB(
        text_chunks=text_chunks, embedder_type="hugging_face", vs_type="chroma_db"
    )
