from typing import Any, List, cast

import streamlit as st

from LLM_UI.custom_types import EmbedderInputTypes, LLMName, VSInputTypes
from LLM_UI.mvvm import UI_VM
from templates.html_template import bot_template, css, user_template

ui_vm_openai_faiss = UI_VM(
    LLMName.chat_open_ai, EmbedderInputTypes.open_ai, VSInputTypes.faiss
)

ui_vm_openai_chroma_db = UI_VM(
    LLMName.chat_open_ai, EmbedderInputTypes.hugging_face, VSInputTypes.faiss
)


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


def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs :books:")

    if "model_1" not in st.session_state:
        st.session_state.model_1 = None
    if "model_2" not in st.session_state:
        st.session_state.model_2 = None

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                ui_vm_openai_faiss.process(cast(List[str], pdf_docs))
                ui_vm_openai_chroma_db.process(cast(List[str], pdf_docs))

                st.session_state.model1 = ui_vm_openai_faiss
                st.session_state.model2 = ui_vm_openai_chroma_db

    box_1, box_2 = st.columns(spec=2, gap="large")
    user_question = st.chat_input(
        "Ask a question about your documents and press Enter:"
    )

    if user_question:
        response_model1 = st.session_state.model1.handle_user_input(user_question)
        response_model2 = st.session_state.model2.handle_user_input(user_question)

        with box_1:
            st.subheader("Chat with OpenAI")
            chat_history = response_model1["chat_history"]
            write_to_output(chat_history)

        with box_2:
            st.subheader("Chat with OpenAI")
            chat_history = response_model2["chat_history"]
            write_to_output(chat_history)


if __name__ == "__main__":
    main()
