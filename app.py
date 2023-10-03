from typing import Any, List, cast

import streamlit as st

from LLM_UI.custom_types import EmbedderInputTypes, LLMName, VSInputTypes
from LLM_UI.mvvm import UI_VM
from templates.html_template import bot_template, css, user_template

params_llama = {
    "model_path": '/home/sar/Documents/LLama/llama.cpp/models/13B/ggml-model-q8_0.gguf',
    "n_ctx": 2048,
    "n_batch": 512,
    "n_threads": 8,
    "n_gpu_layers": 20,
    "temperature": 0.7,
    "seed": 2334242,
    "max_tokens": 200,
    "repeat_penalty": 1.18,
    "top_p": 0.1,
    "top_k": 40,
}


ui_vm_openai_faiss = UI_VM(
    LLMName.chat_open_ai, EmbedderInputTypes.open_ai, VSInputTypes.faiss
)

ui_vm_llama_faiss = UI_VM(
    LLMName.llama_cpp,
    EmbedderInputTypes.open_ai,
    VSInputTypes.faiss,
    params_llm=params_llama,
)
# params_llama = {"model_path": '', "n_ctx": 23, "n_threads": 16, "n_gpu_layers": 20}
params_text_splitter = {"chunk_size": 1000, "chunk_overlap": 200}


def write_to_output(chat_history: List[Any]):
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def set_params_text_splitter(param_name: str, val: int):
    params_text_splitter[param_name] = val


def main():
    st.set_page_config(
        layout="wide", page_title="Chat with multiple PDFs", page_icon=":books:"
    )
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs :books:", divider="rainbow")

    if "model_1" not in st.session_state:
        st.session_state.model_1 = None
    if "model_2" not in st.session_state:
        st.session_state.model_2 = None

    if "disabled" not in st.session_state:
        st.session_state["disabled"] = False

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        st.divider()
        st.subheader("Text splitter configuration")

        chunk_size = st.text_input("Chunk size", placeholder="Enter value")
        chunk_overlap = st.text_input("Chunk overlap", placeholder="Enter value")

        if chunk_size:
            params_text_splitter['chunk_size'] = int(chunk_size)

        if chunk_overlap:
            params_text_splitter['over_lap'] = int(chunk_overlap)

        if st.button("Process"):
            with st.spinner("Processing"):
                ui_vm_openai_faiss = UI_VM(
                    LLMName.chat_open_ai,
                    EmbedderInputTypes.open_ai,
                    VSInputTypes.faiss,
                    params_embedder=params_llama,
                    params_text_splitter=params_text_splitter,
                )

                ui_vm_openai_chroma_db = UI_VM(
                    LLMName.chat_open_ai,
                    EmbedderInputTypes.open_ai,
                    VSInputTypes.faiss,
                    params_text_splitter=params_text_splitter,
                )

                ui_vm_openai_faiss.process(cast(List[str], pdf_docs))
                ui_vm_llama_faiss.process(cast(List[str], pdf_docs))

                st.session_state.model1 = ui_vm_openai_faiss
                st.session_state.model2 = ui_vm_llama_faiss

    box_1, box_2, box_3 = st.columns((1.5, 1, 1), gap="large")
    user_question = st.chat_input(
        "Ask a question about your documents and press Enter:"
    )

    if user_question:
        response_model1 = st.session_state.model1.handle_user_input(user_question)
        response_model2 = st.session_state.model2.handle_user_input(user_question)

        with box_2:
            st.subheader("Chat with OpenAI")
            chat_history = response_model1["chat_history"]
            write_to_output(chat_history)

        with box_3:
            st.subheader("Chat with Llama")
            chat_history = response_model2["chat_history"]
            write_to_output(chat_history)

        with box_1:
            st.subheader("Result vector database")


if __name__ == "__main__":
    main()
