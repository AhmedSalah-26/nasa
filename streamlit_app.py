import streamlit as st
import requests
import json

# Set page config
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #f5f5f5;
}
.css-1d391kg {
    padding: 2rem;
}
.stButton>button {
    background-color: #2e7d32;
    color: white;
    border-radius: 5px;
    padding: 0.5rem 1rem;
    border: none;
}
.stButton>button:hover {
    background-color: #1b5e20;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("📚 PDF Chat Assistant")
st.markdown("---")

# File upload
st.subheader("1. Upload your PDF file")
upload_placeholder = st.empty()
with upload_placeholder.container():
    uploaded_file = st.file_uploader("", type="pdf")

# Process PDF
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        response = requests.post("http://localhost:8000/upload-pdf/", files=files)
        if response.status_code == 200:
            st.success("PDF processed successfully!")
            upload_placeholder.empty()
            
            # Chat interface
            st.subheader("2. Ask questions about your PDF")
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Ask a question about your PDF"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = requests.post(
                            "http://localhost:8000/chat/",
                            json={"question": prompt}
                        )
                        if response.status_code == 200:
                            answer = response.json()["answer"]
                            st.markdown(answer)
                            # Add assistant message to chat history
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        else:
                            st.error("Error getting response from the server.")
        else:
            st.error("Error processing PDF.")

# Instructions
with st.sidebar:
    st.markdown("### How to use")
    st.markdown("""
    1. Upload your PDF file using the upload button
    2. Wait for the processing to complete
    3. Ask questions about the content of your PDF
    4. The AI will respond based on the information in your document
    """)
    
    st.markdown("### About")
    st.markdown("""
    This application uses AI to help you interact with your PDF documents.
    Upload a PDF and ask questions about its content in natural language.
    """)