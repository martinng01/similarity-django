from unittest import TestLoader
from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from .build_vectordb import build_database, CHROMA_DB_DIRECTORY

from langchain_chroma import Chroma

load_dotenv(override=True)

CHROMA_DB_DIRECTORY = 'chroma_db/similarity'
sbert_embed = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese")


def database_exists():
    print(f'Database exists: {os.path.exists(CHROMA_DB_DIRECTORY)}')
    return os.path.exists(CHROMA_DB_DIRECTORY)


def setup_backend():
    if not database_exists():
        build_database()


def build_database():
    # Options:
    # shibing624/text2vec-base-chinese
    # uer/sbert-base-chinese-nli
    sbert_embed = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"], chunk_size=1, chunk_overlap=0, keep_separator=False
    )

    inf_loader = TextLoader("data/informal.txt")
    inf_docs = inf_loader.load()
    inf_splits = text_splitter.split_documents(inf_docs)

    Chroma.from_documents(
        documents=inf_splits[:5000],  # Array split here for testing
        collection_name="informal",
        embedding=sbert_embed,
        persist_directory=CHROMA_DB_DIRECTORY
    )


def get_similar_sentences(sentence, k=3):
    db = Chroma(
        collection_name="informal",
        embedding_function=sbert_embed,
        persist_directory=CHROMA_DB_DIRECTORY
    )
    return [d.page_content for d in db.similarity_search(sentence, k=k)]
