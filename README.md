Here’s your text fully translated into English:

---

# PDF Chat Application

A web application that uses artificial intelligence to analyze PDF files and answer questions related to their content.

## Features

* Upload and analyze PDF files
* Extract text from files
* Interactive chat interface
* Intelligent answers to questions
* Beautiful and user-friendly interface

## Requirements

* Python 3.8 or newer
* Libraries listed in `requirements.txt`

## Installation

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Add your HuggingFace API key to environment variables:

```bash
set HUGGINGFACE_API_KEY=your-api-key
```

## Running the Application

1. Start the FastAPI server:

```bash
uvicorn main:app --reload
```

2. In a new terminal window, start the user interface:

```bash
streamlit run streamlit_app.py
```

3. Open your browser at: [http://localhost:8501](http://localhost:8501)

## How to Use

1. Upload a PDF file using the upload button
2. Wait for the file to be processed
3. Ask questions about the content of the file
4. The system will respond based on the information in the file

---

If you want, I can also **convert it into a fully polished README file** ready for GitHub. Do you want me to do that?
