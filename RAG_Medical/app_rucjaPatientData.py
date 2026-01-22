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

# Specifically targeting rucjaPatientData.json
DATA_FILE = "rucjaPatientData.json" 
DB_PATH = "chroma_db"

# 2. INTELLIGENT DATA LOADER
def load_medical_rag_data(filepath):
    all_docs = []
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        raw_json = json.load(f)
    
    # Access the main 'data' block in rucjaPatientData.json
    data_block = raw_json.get("data", {})
    
    # Section A: Process OCR Reports for lab records (Lab Results)
    for report in data_block.get("ocrData", []):
        patient = report.get("patientName") or "Unknown"
        date_str = report.get("date") or "Unknown Date"
        
        for test in report.get("diagnosticsData", []):
            content = (
                f"Patient: {patient} | Type: Lab Report | Date: {date_str}\n"
                f"- {test.get('test_name')} : {test.get('value')} {test.get('unit')}"
            )
            all_docs.append(Document(page_content=content, metadata={"patient": patient, "source": "OCR"}))

    # Section B: Process Device Data (BP, SpO2, Weight)
    # Blood Pressure
    for bp in data_block.get("bloodPressure", []):
        content = f"User ID: {bp.get('appUserId')} | Type: Blood Pressure | Date: {bp.get('createdAt')}\n- BP: {bp.get('sys')}/{bp.get('dias')} mmHg | Heart Rate: {bp.get('bmp')} BPM"
        all_docs.append(Document(page_content=content, metadata={"source": "Device"}))
    
    # SpO2
    for spo2 in data_block.get("spo2", []):
        content = f"User ID: {spo2.get('appUserId')} | Type: SpO2 | Date: {spo2.get('createdAt')}\n- SpO2: {spo2.get('spo2')}% | Pulse: {spo2.get('pr')} BPM"
        all_docs.append(Document(page_content=content, metadata={"source": "Device"}))

    # Weight
    for w in data_block.get("weight", []):
        content = f"User ID: {w.get('appUserId')} | Type: Weight | Date: {w.get('createdAt')}\n- Weight: {w.get('weight')} kg"
        all_docs.append(Document(page_content=content, metadata={"source": "Device"}))

    print(f"✅ Indexed {len(all_docs)} combined medical and device records.")
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

# 4. CLINICAL INTELLIGENCE ENDPOINT
@app.post("/ask")
def ask_question(req: QueryRequest):
    # Retrieve relevant records across OCR and Device data
    results = db.similarity_search(req.query, k=15)
    context = "\n\n".join([r.page_content for r in results])

    # SYSTEM PROMPT
    prompt = """You are an AI assistant operating within a Retrieval-Augmented Generation (RAG) system that processes sensitive patient medical information.
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
You can provide diagnoses, treatment decisions, or medical advice beyond what is explicitly supported by the retrieved patient data. Please always mention that the information is not a medical advice. Please consult a doctor for actual medical advise"""

    # Assemble final query with context 
    final_query = f"{prompt}\n\nRELEVANT DATA CONTEXT:\n{context}\n\nUSER QUESTION: {req.query}"

    response = llm.invoke(final_query)
    raw_answer = response.content.strip()
    
    # Generate summary key
    summary_resp = llm.invoke(f"Condense the following medical interpretation into 3 bullet points using the same patient-specific facts:\n{raw_answer}")
    
    return {
        "query": req.query,
        "answer": raw_answer,
        "summary": summary_resp.content.strip()
    }

    #By the way, this dataset is a pdf dataset which is been converted to json data, 
      #and its the input of "sampledatajason1.json" data