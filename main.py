"""
PDF/URL/PMC Q&A API — Gemini API extraction (Background/Results/Conclusion)
Requirements:
pip install fastapi uvicorn langchain langchain-community faiss-cpu pypdf torch aiohttp beautifulsoup4 biopython requests
"""

import os
import tempfile
import json
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import uvicorn
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    raise ImportError("❌ pypdf library is missing. Install it with: pip install pypdf")

from langchain_community.embeddings import HuggingFaceEmbeddings
from Bio import Entrez

# ===============================
# Email for NCBI Entrez
# ===============================
Entrez.email = "your_email@example.com"

# ===============================
# FastAPI settings
# ===============================
app = FastAPI(title="PDF/URL/PMC Gemini API", version="2.5")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ===============================
# Gemini API configuration
# ===============================
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyAcn9IK3b70MRqOqLcWfiZzc3j3aq3_gf4"


# ===============================
# Vectorstore & embeddings
# ===============================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = None

# ===============================
# Helper functions
# ===============================
async def fetch_url_content(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    import aiohttp
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to fetch URL {url} (status {resp.status})")
            html = await resp.text()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style"]): 
        tag.decompose()
    return soup.get_text(separator="\n")

def fetch_pmc_article(pmc_id: str) -> str:
    handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="full", retmode="text")
    text = handle.read()
    handle.close()
    if not text.strip():
        raise Exception("PMC returned empty text")
    return text

def split_text_into_docs(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_text(text)

def vectorstore_from_docs(docs):
    global vectorstore
    vectorstore = FAISS.from_texts(docs, embeddings)

# ===============================
# Request models
# ===============================
class URLRequest(BaseModel):
    url: HttpUrl

class PMCRequest(BaseModel):
    pmc_id: str

class QuestionRequest(BaseModel):
    question: str
    k: int = 5

# ===============================
# Endpoints
# ===============================
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        docs_split = split_text_into_docs("\n".join([d.page_content for d in docs]))
        vectorstore_from_docs(docs_split)
        os.unlink(tmp_path)
        return {"status":"success","num_chunks":len(docs_split)}
    except Exception as e:
        return {"status":"error","message":str(e)}

@app.post("/upload-url/")
async def upload_url(req: URLRequest):
    try:
        raw_text = await fetch_url_content(req.url)
        docs_split = split_text_into_docs(raw_text)
        vectorstore_from_docs(docs_split)
        return {"status":"success","num_chunks":len(docs_split)}
    except Exception as e:
        return {"status":"error","message":str(e)}

@app.post("/upload-pmc/")
async def upload_pmc(req: PMCRequest):
    try:
        raw_text = fetch_pmc_article(req.pmc_id)
        docs_split = split_text_into_docs(raw_text)
        vectorstore_from_docs(docs_split)
        return {"status":"success","num_chunks":len(docs_split)}
    except Exception as e:
        return {"status":"error","message":str(e)}

# ===============================
# Ask endpoint — Gemini Q&A
# ===============================
@app.post("/ask/")
async def ask_question(req: QuestionRequest):
    if vectorstore is None:
        return {"status": "error", "message":"No document uploaded yet"}
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": req.k})
        docs = retriever.get_relevant_documents(req.question)
        context = "\n\n".join([d.page_content for d in docs])
        print("🔹 Sending text to Gemini API...")

        prompt = (
            "Read the following text and extract sections precisely: "
            "Background, Results, Conclusion. Respond only in JSON format with these keys.\n\n" + context
        )
        payload = {"contents":[{"parts":[{"text": prompt}]}]}
        response = requests.post(GEMINI_URL, json=payload, timeout=60)
        merged_result = {"Background": "", "Results": "", "Conclusion": ""}

        if response.status_code == 200:
            result = response.json()
            candidates = result.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    text_output = parts[0].get("text", "").strip()

                    # Remove markdown code fences
                    if text_output.startswith("```"):
                        lines = text_output.split("\n")
                        text_output = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                    
                    text_output = text_output.strip()

                    try:
                        # Parse multiple times for nested JSON strings
                        json_result = text_output
                        max_iterations = 5
                        iteration = 0
                        
                        while isinstance(json_result, str) and iteration < max_iterations:
                            json_result = json.loads(json_result)
                            iteration += 1
                        
                        if isinstance(json_result, dict):
                            for key in merged_result:
                                merged_result[key] = json_result.get(key, "").strip()
                        else:
                            merged_result["Background"] = str(json_result)
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON parse error: {e}")
                        merged_result["Background"] = text_output
        else:
            merged_result["Background"] = f"[Gemini API failed: status {response.status_code}]"

        print("🔹 Gemini API response received.")
        print(f"🔹 Extracted sections: Background={len(merged_result['Background'])} chars, Results={len(merged_result['Results'])} chars, Conclusion={len(merged_result['Conclusion'])} chars")
        
        return {"status": "success", "result": merged_result, "retrieved_chunks": len(docs)}

    except Exception as e:
        return {"status":"error","message":str(e)}

# ===============================
# Run server
# ===============================
if __name__ == "__main__":
    print("🚀 Starting server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
