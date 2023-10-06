from typing import Any, List, cast

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler

from LLM_UI.custom_types import EmbedderInputTypes, LLMName, VSInputTypes
from LLM_UI.mvvm import UI_VM
from templates.html_template import bot_template, css, user_template


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method
        self.new_sentence = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.new_sentence += token

        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(
                bot_template.replace("{{MSG}}", self.text), unsafe_allow_html=True
            )
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

    def on_llm_end(self, response, **kwargs) -> None:
        self.text = ""


params_llama = {
    "model_path": '/home/sar/Documents/LLama/llama.cpp/models/13B/ggml-model-q8_0.gguf',
    "n_ctx": 2048,
    "n_batch": 512,
    "n_threads": 8,
    "n_gpu_layers": 20,
    "temperature": 0.7,
    "seed": 2334242,
    "max_tokens": 256,
    "repeat_penalty": 1.2,
    "top_p": 0.1,
    "top_k": 40,
    "streaming": True,
    "callbacks": None,
}

params_open_ai = {'streaming': True, 'callbacks': None}
# params_llama = {'streaming': True, 'callbacks': None}
params_text_splitter = {"chunk_size": 500, "chunk_overlap": 75}


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

    # chat_box = st.empty()
    box_1, box_2, box_3 = st.columns((1.5, 1, 1), gap="large")

    with box_1:
        st.subheader("Vector DB search result")

    with box_2:
        st.subheader("Chat with OpenAI")
        chat_box_openai = st.empty()

    with box_3:
        st.subheader("Chat with LLama")
        chat_box_llama = st.empty()

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
            params_text_splitter['chunk_overlap'] = int(chunk_overlap)

        if st.button("Process"):
            with st.spinner("Processing"):
                stream_handler1 = StreamHandler(chat_box_openai, display_method='write')
                stream_handler2 = StreamHandler(chat_box_llama, display_method='write')

                params_open_ai["callbacks"] = [stream_handler1]
                params_llama["callbacks"] = [stream_handler2]

                ui_vm_openai_faiss = UI_VM(
                    LLMName.chat_open_ai,
                    EmbedderInputTypes.open_ai,
                    VSInputTypes.faiss,
                    params_llm=params_open_ai,
                    params_text_splitter=params_text_splitter,
                )

                ui_vm_llama_faiss = UI_VM(
                    LLMName.llama_cpp,
                    EmbedderInputTypes.open_ai,
                    VSInputTypes.faiss,
                    params_llm=params_llama,
                    params_text_splitter=params_text_splitter,
                )

                ui_vm_openai_faiss.process(cast(List[str], pdf_docs))
                ui_vm_llama_faiss.process(cast(List[str], pdf_docs))

                st.session_state.model1 = ui_vm_openai_faiss
                st.session_state.model2 = ui_vm_llama_faiss

    user_question = st.chat_input(
        "Ask a question about your documents and press Enter:"
    )

    if user_question:
        with box_2:
            st.write(
                user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True
            )

        with box_3:
            st.write(
                user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True
            )

        response_model1 = st.session_state.model1.handle_user_input(user_question)
        response_model2 = st.session_state.model2.handle_user_input(user_question)

        with box_1:
            db_search_result = response_model1["source_documents"]

            for result in db_search_result:
                st.write(result.page_content, unsafe_allow_html=True)
                st.divider()


if __name__ == "__main__":
    main()
