import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_NAME")

DATA_FILE = "sampleData.json" 
DB_PATH = "chroma_db"

# 2. INTELLIGENT DATA LOADER
def load_medical_rag_data(filepath):
    all_docs = []
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        raw_json = json.load(f)
    
    data_content = raw_json.get("data", {})
    
    # Normalizing medical terms for better retrieval (WBC/TLC, etc.)
    for report in data_content.get("ocrData", []):
        date_val = report.get("createdAt", "Unknown Date")
        
        for test in report.get("diagnosticsData", []):
            t_name = test.get('test_name', '')
            # Mapping synonyms directly into the searchable text
            if "total leukocyte count" in t_name.lower() or "tlc" in t_name.lower():
                t_name = f"{t_name} (WBC Total Leukocyte Count)"
            
            content = (
                f"Date: {date_val}\n"
                f"- {t_name} : {test.get('value')} {test.get('unit')}\n"
                f"Confidence: {test.get('confidence')}%"
            )
            all_docs.append(Document(page_content=content))

    print(f"✅ Indexed {len(all_docs)} diagnostic entries for RAG.")
    return all_docs


# 3. GLOBAL INITIALIZATION
all_docs = load_medical_rag_data(DATA_FILE)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_KEY,
    openai_api_version=AZURE_VERSION,
    azure_deployment="text-embedding-ada-002"
)

if os.path.exists(DB_PATH):
    import shutil
    shutil.rmtree(DB_PATH)

db = Chroma.from_documents(all_docs, embeddings, persist_directory=DB_PATH)

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    deployment_name=AZURE_DEPLOYMENT,
    api_key=AZURE_KEY,
    openai_api_version=AZURE_VERSION,
    temperature=0
)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# 4. SPECIFIC INTELLIGENCE ENDPOINT
@app.post("/ask")
def ask_question(req: QueryRequest):
    # Retrieve relevant records
    results = db.similarity_search(req.query, k=10)
    context = "\n\n".join([r.page_content for r in results])

    # HR'S EXACT SYSTEM PROMPT + REINFORCED CLEANUP
    system_prompt = """You are an AI assistant operating within a Retrieval-Augmented Generation (RAG) system that processes sensitive patient medical information.
The retrieved data consists of:
Patient reports generated from medical devices
Digitized and scanned medical report documents
Your task is to:
Use only the retrieved patient-specific data provided in the context
Interpret the information accurately and conservatively
Generate clear, contextual, and medically relevant answers to user queries
Avoid speculation and do not introduce information that is not present in the retrieved data
Ensure that all responses:
Respect patient privacy and data sensitivity
Are grounded strictly in the supplied medical records
Clearly indicate uncertainty or missing information when the data is insufficient
You can provide diagnoses, treatment decisions, or medical advice beyond what is explicitly supported by the retrieved patient data. Please always mention that the information is not a medical advice. Please consult a doctor for actual medical advise.
IMPORTANT: THE OUTPUT MUST BE PLAIN TEXT ONLY. DO NOT USE MARKDOWN (NO **). DO NOT USE NEWLINES (NO \\n). ALL CONTENT MUST BE IN A SINGLE, CLEAN PARAGRAPH."""

    final_prompt = f"{system_prompt}\n\nRetrieved Context:\n{context}\n\nUser Query: {req.query}"

    response = llm.invoke(final_prompt)
    
    # NEW LOGIC: This double-clean ensures literal \n and actual newlines are removed
    raw_content = response.content.strip().replace("**", "")
    # Removes both physical newlines and literal '\n' characters
    clean_answer = " ".join(raw_content.splitlines()).replace("\\n", " ")

    # Apply the same clean logic to the summary
    summary_prompt = f"Condense this into a 1-sentence plain text summary: {clean_answer}"
    summary_resp = llm.invoke(summary_prompt)
    clean_summary = " ".join(summary_resp.content.strip().replace("**", "").splitlines()).replace("\\n", " ")
    
    return {
        "query": req.query,
        "answer": clean_answer,
        "summary": clean_summary
    }
