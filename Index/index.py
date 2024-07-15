

### Build Index

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Blob
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

### from langchain_cohere import CohereEmbeddings

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

# Set embeddings
embd = OpenAIEmbeddings()

# Docs to index
paths = [
    "/Users/Zoo/Library/Mobile Documents/com~apple~Keynote/Documents/Resume_2024.pdf",
    # "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    # "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load
docs = [PyPDFLoader(path).load() for path in paths]
docs_list = [item for sublist in docs for item in sublist]

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorstore
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embd,
)
retriever = vectorstore.as_retriever()