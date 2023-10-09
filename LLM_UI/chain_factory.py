from typing import Any, Dict

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub, LlamaCpp, OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate

from LLM_UI.custom_types import LLMName
from LLM_UI.embedder import VectorDB

llm_map: Dict[str, Any] = {
    "hugging_face_hub": HuggingFaceHub,
    "llama_cpp": LlamaCpp,
    "open_ai": OpenAI,
    "chat_open_ai": ChatOpenAI,
}

_TEMPLATE_PROMPT = """ \
<s>[INST] <<SYS>>
Je bent een behulpzame, respectvolle en eerlijke assistant. Antwoord altijd zo behulpzaam mogelijk.
Je antwoorden mogen geen schadelijke, onethische, racistische, seksistische, gevaarlijke of illegale inhoud bevatten. Zorg ervoor dat je antwoorden sociaal onbevooroordeeld en positief.

Gebruik de volgende stukjes context om de vraag van de gebruiker te beantwoorden. De antwoorden moeten in het Nederlands zijn.
Also een vraag nergens op slaat of feitelijk niet coherent is, leg dan uit waarom in plaats van iets niet correct te antwoorden. Also je het antwoord op een vraag niet weet, deel dan geen onjuiste informative.
<</SYS>>
context : {context}
question: {question} [/INST]
"""


def chain_factory(llm_name: LLMName, vector_db: VectorDB, **kwargs):
    llm = llm_map[llm_name](**kwargs)
    memory = ConversationBufferMemory(
        memory_key="chat_history", output_key="answer", return_messages=True
    )

    chat_prompt = PromptTemplate.from_template(_TEMPLATE_PROMPT)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.get_db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 3},
        ),
        return_generated_question=True,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": chat_prompt},
    )
    return conversation_chain
