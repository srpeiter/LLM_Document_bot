import lorem as lm

from LLM_UI.custom_types import EmbedderInputTypes, VSInputTypes
from LLM_UI.embedder import VectorDB, _embedder_map, _vector_store_map


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
