import os
from dotenv import load_dotenv, find_dotenv
from utils.validators import validate_inputs
from utils.date_parser import extract_date
from utils.load_documents import load_and_embed_documents
from utils.chatbot_chain import build_chatbot_chain
from agent_tools.book_appointment_tool import appointment_agent
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
import streamlit as st

_ = load_dotenv(find_dotenv())
gemini_api_key = os.environ["GOOGLE_API_KEY"]

# Streamlit page setup
st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("📄 Document QA Chatbot with Appointment Booking")

# Session state setup
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'form_active' not in st.session_state:
    st.session_state.form_active = False
if 'form_data' not in st.session_state:
    st.session_state.form_data = {'name': '', 'phone': '', 'email': '', 'date': ''}
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None

# Upload and embed documents 
@st.cache_data(show_spinner=True)
def process_docs(files):
    docs = []
    for f in files:
        path = os.path.join("uploaded_docs", f.name)
        with open(path, "wb") as out:
            out.write(f.read())
        docs.extend(load_and_embed_documents(path))
    return docs

uploaded_files = st.file_uploader("Upload Documents (PDF, TXT, DOCX, CSV, HTML, Excel)", accept_multiple_files=True)


if uploaded_files and not st.session_state.qa_chain:
    os.makedirs("uploaded_docs", exist_ok=True)
    documents = process_docs(uploaded_files)
    qa_chain = build_chatbot_chain(documents)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    agent_executor = initialize_agent([appointment_agent], llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

    st.session_state.qa_chain = qa_chain
    st.session_state.agent_executor = agent_executor

if 'form_step' not in st.session_state:
    st.session_state.form_step = 0

def appointment_form(user_reply):
    steps = ['name', 'phone', 'email', 'date']
    prompts = {
        'name': "What's your name?",
        'phone': "What's your phone number?",
        'email': "What's your email address?",
        'date': "When would you like to book the appointment? (e.g. next Monday)"
    }

    if 'form_step' not in st.session_state:
        st.session_state.form_step = 0

    if user_reply:
        current_field = steps[st.session_state.form_step]
        if current_field == 'date':
            parsed_date = extract_date(user_reply)
            if parsed_date:
                st.session_state.form_data['date'] = parsed_date
                st.session_state.form_step += 1
            else:
                st.chat_message("assistant").markdown("Sorry, I couldn't extract a valid date. Try again in this format: `YYYY-MM-DD` or like 'next Friday'.")
                return
        else:
            st.session_state.form_data[current_field] = user_reply
            st.session_state.form_step += 1

    if st.session_state.form_step < len(steps):
        next_field = steps[st.session_state.form_step]
        st.chat_message("assistant").markdown(prompts[next_field])
    else:
        # All fields collected — validate
        valid, msg = validate_inputs(
            st.session_state.form_data['name'],
            st.session_state.form_data['phone'],
            st.session_state.form_data['email']
        )

        if not valid:
            st.chat_message("assistant").markdown(f"{msg}")
            st.session_state.form_step = 0  
            st.session_state.form_data = {'name': '', 'phone': '', 'email': '', 'date': ''}
            st.session_state.form_restart_required = True
            return
        else:
            confirmation = appointment_agent.invoke(st.session_state.form_data)
            st.chat_message("assistant").markdown(f" {confirmation}")
            st.session_state.form_active = False
            st.session_state.form_data = {'name': '', 'phone': '', 'email': '', 'date': ''}
            st.session_state.form_step = 0
            st.session_state.form_restart_required = False
            return

user_input = st.chat_input("Ask a question or respond to the form:")

form_should_prompt = False

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.form_active:
        if st.session_state.form_restart_required: 
            st.chat_message("assistant").markdown("What's your name?")
            st.session_state.form_restart_required = False
        else:
            appointment_form(user_input)
    elif "call me" in user_input.lower() or "book appointment" in user_input.lower():
        st.session_state.form_active = True
        st.session_state.form_step = 0
        st.session_state.form_data = {'name': '', 'phone': '', 'email': '', 'date': ''}
        st.session_state.form_restart_required = False

        with st.chat_message("assistant"):
            st.markdown("### Let's book your appointment")
            st.markdown("What's your name?")
    else:
        with st.chat_message("assistant"):
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", streaming=True)
            streamed_chain = ConversationalRetrievalChain.from_llm(llm, st.session_state.qa_chain.retriever)
            result = streamed_chain.invoke({"question": user_input, "chat_history": st.session_state.chat_history})
            st.markdown(result['answer'])
            st.session_state.chat_history.append((user_input, result['answer']))

if st.session_state.form_active and not user_input:
    appointment_form("")