from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_documents(urls: list, chunk_size=100, chunk_overlap=50):
    docs = [WebBaseLoader(url).load() for url in urls]
    flat_docs = [doc for sublist in docs for doc in sublist]
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(flat_docs)