from PyPDF2 import PdfReader
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


class UI_VM(object):
    def __init__(self):
        self.text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

        embedding_model = HuggingFaceEmbeddings()

        self.db = Chroma(embedding_function=embedding_model)

    def import_PDF(self, file: str):
        pdf = PdfReader(file)

        text: str = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text

    def _get_text_chunks(self, text: str):
        return self.text_splitter.split_text(text)

    def db_vector_insert(self, chunks: str):
        pass

        # self.db.add_texts(self._get_text_chunks)
