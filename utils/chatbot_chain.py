from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from agent_tools.book_appointment_tool import appointment_agent

def build_chatbot_chain(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",streaming=True)
    return ConversationalRetrievalChain.from_llm(llm, retriever)

def build_agent_with_tools():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    tools = [appointment_agent]

    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,  # Works with Gemini too
        verbose=True,
        handle_parsing_errors=True,
    )

    return agent_executor