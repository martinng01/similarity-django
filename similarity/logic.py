from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(override=True)

CHROMA_DB_DIRECTORY = 'chroma_db/similarity'

db = None
num_sentences_processed = 0


def database_exists():
    print(f'Database exists: {os.path.exists(CHROMA_DB_DIRECTORY)}')
    return os.path.exists(CHROMA_DB_DIRECTORY)


def setup_backend():
    if not database_exists():
        build_database()


def build_database():
    sbert_embed = HuggingFaceEmbeddings(
        model_name="uer/sbert-base-chinese-nli")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"], chunk_size=1, chunk_overlap=0, keep_separator=False
    )

    inf_loader = TextLoader("similarity/data/informal.txt")
    inf_docs = inf_loader.load()
    inf_splits = text_splitter.split_documents(inf_docs)

    global db
    db = Chroma.from_documents(
        documents=inf_splits,  # Array split here for testing
        collection_name="informal",
        embedding=sbert_embed,
        persist_directory=CHROMA_DB_DIRECTORY
    )


def get_similar_sentences(sentence, k=3):
    return [d.page_content for d in db.similarity_search(sentence, k=k)]
