from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.tools.retriever import create_retriever_tool

def setup_vectorstore(documents, collection_name="rag-chroma-happy", model="llama3.2"):
    embedding = OllamaEmbeddings(model=model)
    vectorstore = Chroma.from_documents(documents, collection_name=collection_name, embedding=embedding)
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        name="retrieve_blog_posts",
        description="Search and return information about happiness in the blog posts.",
    )
    return retriever_tool