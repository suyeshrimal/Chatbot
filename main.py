import os
from dotenv import load_dotenv,find_dotenv

from utils.load_documents import load_and_embed_documents
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

_ = load_dotenv(find_dotenv())

gemini_api_key =  os.environ["GOOGLE_API_KEY"]

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. Use the following context to answer the question at the end.
    
    Context:
    {context}
    
    Question: {input}
    
    Answer:
    """
)

file_path = "data\MyResume.pdf" 
db = load_and_embed_documents(file_path)
retriever = db.as_retriever()

llm = ChatGoogleGenerativeAI(
    model = "gemini-1.5-flash",
)

combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

query = "What is the purpose of this document?"
response = retrieval_chain.invoke({"input":"Explain what the document is about?"})

print("Answer:", response)