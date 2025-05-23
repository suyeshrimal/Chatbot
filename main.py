import os
from dotenv import load_dotenv,find_dotenv

_ = load_dotenv(find_dotenv())

gemini_api_key =  os.environ["GOOGLE_API_KEY"]

import streamlit as st
from utils.load_documents import load_and_embed_documents
from utils.chatbot_chain import build_chatbot_chain
from agent_tools.book_appointment_tool import appointment_agent

st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("📄 Document QA Chatbot with Callback Support")

uploaded_files = st.file_uploader("Upload Documents (PDF, TXT, DOCX, CSV, HTML, Excel)", accept_multiple_files=True)
query = st.text_input("Ask a question based on your documents:")

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

    qa_chain = build_chatbot_chain(all_docs)

    if query:
        if "call me" in query.lower() or "book appointment" in query.lower():
            with st.form("callback_form"):
                name = st.text_input("Enter your name")
                phone = st.text_input("Enter your phone number")
                email = st.text_input("Enter your email")
                date_input = st.text_input("Preferred appointment date (e.g., next Monday)")
                submitted = st.form_submit_button("Submit")

                if submitted:
                    from utils.validators import validate_inputs
                    from utils.date_parser import extract_date
                    valid, msg = validate_inputs(name, phone, email)
                    if not valid:
                        st.error(msg)
                    else:
                        full_date = extract_date(date_input)
                        if not full_date:
                            st.error("Could not extract a valid date in YYYY-MM-DD format.")
                        else:
                            confirmation = appointment_agent(name, phone, email, full_date)
                            st.success(f"Appointment booked for {name} on {full_date}. Confirmation: {confirmation}")
        else:
            result = qa_chain.invoke({"question": query, "chat_history": st.session_state.chat_history})
            st.session_state.chat_history.append((query, result['answer']))
            st.markdown(f"**Answer:** {result['answer']}")