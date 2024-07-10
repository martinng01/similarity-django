"""
Run this script to build the database for the similarity search.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma

CHROMA_DB_DIRECTORY = 'chroma_db/similarity'


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

    global db
    db = Chroma.from_documents(
        documents=inf_splits[:5000],  # Array split here for testing
        collection_name="informal",
        embedding=sbert_embed,
        persist_directory=CHROMA_DB_DIRECTORY
    )


if __name__ == '__main__':
    build_database()
