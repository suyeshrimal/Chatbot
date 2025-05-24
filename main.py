import os
from dotenv import load_dotenv,find_dotenv
from utils.validators import validate_inputs
from utils.date_parser import extract_date
_ = load_dotenv(find_dotenv())

gemini_api_key =  os.environ["GOOGLE_API_KEY"]

import streamlit as st
from utils.load_documents import load_and_embed_documents
from utils.chatbot_chain import build_chatbot_chain
from agent_tools.book_appointment_tool import appointment_agent
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("📄 Document QA Chatbot with Callback Support")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'form_active' not in st.session_state:
    st.session_state.form_active = False
if 'form_step' not in st.session_state:
    st.session_state.form_step = 0
if 'form_data' not in st.session_state:
    st.session_state.form_data = {'name': '', 'phone': '', 'email': '', 'date': ''}

uploaded_files = st.file_uploader("Upload Documents (PDF, TXT, DOCX, CSV, HTML, Excel)", accept_multiple_files=True)
query = st.chat_input("Ask a question based on your documents:")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:
    UPLOAD_FOLDER = "uploaded_docs"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    all_docs = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        all_docs.extend(load_and_embed_documents(file_path))

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    tools = [appointment_agent]
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )

    qa_chain = build_chatbot_chain(all_docs)

    if st.session_state.form_active:
        step = st.session_state.form_step
        form_data = st.session_state.form_data

        if step == 0:
            name = st.text_input("What is your name?", value=form_data['name'])
            if name:
                st.session_state.form_data['name'] = name
                st.session_state.form_step = 1
                st.rerun()

        elif step == 1:
            email = st.text_input("Thanks! What is your email?", value=form_data['email'])
            if email:
                st.session_state.form_data['email'] = email
                st.session_state.form_step = 2
                st.rerun()

        elif step == 2:
            phone = st.text_input("And your phone number?", value=form_data['phone'])
            if phone:
                st.session_state.form_data['phone'] = phone
                st.session_state.form_step = 3
                st.rerun()

        elif step == 3:
            date_input = st.text_input("Preferred appointment date? (e.g. next Monday)", value=form_data['date'])
            if date_input:
                full_date = extract_date(date_input)
                if not full_date:
                    st.error("Couldn't extract a valid date in YYYY-MM-DD format.")
                else:
                    st.session_state.form_data['date'] = full_date
                    st.session_state.form_step = 4
                    st.rerun()

        elif step == 4:
            # Validate inputs
            name = form_data['name']
            phone = form_data['phone']
            email = form_data['email']
            full_date = form_data['date']
            valid, msg = validate_inputs(name, phone, email)
            if not valid:
                st.error(msg)
            else:
                confirmation = appointment_agent.invoke({
                                            "name": name,
                                            "phone": phone,
                                            "email": email,
                                            "date": full_date
                                            })
                st.success(f"✅ {confirmation}")
            # Reset form
            st.session_state.form_active = False
            st.session_state.form_step = 0
            st.session_state.form_data = {'name': '', 'phone': '', 'email': '', 'date': ''}

    elif query:
        if "call me" in query.lower() or "book appointment" in query.lower():
            st.session_state.form_active = True
            st.session_state.form_step = 0
            st.rerun()
        else:
            # Stream response like ChatGPT
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                stream_handler = StreamlitCallbackHandler(st.container())
                streaming_llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", streaming=True, callbacks=[stream_handler]
                )
                streamed_chain = ConversationalRetrievalChain.from_llm(streaming_llm, qa_chain.retriever)
                result = streamed_chain.invoke({"question": query, "chat_history": st.session_state.chat_history})
                answer = result.get('answer') or result.get('result')  # fallback if key varies

                if answer:
                    st.markdown(answer)
                    st.session_state.chat_history.append((query, answer))
                else:
                    st.error("⚠️ No response was generated.")