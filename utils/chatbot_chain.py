from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI


def build_chatbot_chain(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    return ConversationalRetrievalChain.from_llm(llm, retriever)