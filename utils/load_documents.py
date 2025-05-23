from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredHTMLLoader,  # This one is fairly light
    UnstructuredExcelLoader  # Optional – consider replacing
)
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def choose_loader(file_path):
    extension = Path(file_path).suffix.lower()
    if extension == ".txt":
        return TextLoader(file_path)
    elif extension == ".pdf":
        return PyPDFLoader(file_path)
    elif extension == ".html":
        return UnstructuredHTMLLoader(file_path)
    elif extension in [".doc", ".docx"]:
        return Docx2txtLoader(file_path)
    elif extension in [".csv"]:
        return CSVLoader(file_path)
    elif extension in [".xlsx", ".xls"]:
        return UnstructuredExcelLoader(file_path)
    else:
        raise ValueError(f"Unsupported File Type: {extension}")
    

def load_and_embed_documents(file_path):
    loader = choose_loader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_documents(split_docs, embedding=embeddings)
    return db