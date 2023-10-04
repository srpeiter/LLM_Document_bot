from typing import Any, Dict

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub, LlamaCpp, OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

from LLM_UI.custom_types import LLMName
from LLM_UI.embedder import VectorDB

llm_map: Dict[str, Any] = {
    "hugging_face_hub": HuggingFaceHub,
    "llama_cpp": LlamaCpp,
    "open_ai": OpenAI,
    "chat_open_ai": ChatOpenAI,
}


def chain_factory(llm_name: LLMName, vector_db: VectorDB, **kwargs):
    llm = llm_map[llm_name](**kwargs)
    memory = ConversationBufferMemory(
        memory_key="chat_history", output_key="answer", return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.get_db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 3},
        ),
        return_generated_question=True,
        return_source_documents=True,
        verbose=True,
    )
    return conversation_chain
