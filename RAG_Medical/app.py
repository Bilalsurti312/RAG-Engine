import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# -----------------------
# 1. ENV SETUP
# -----------------------
load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_NAME")

# Focusing on sampleData.json per your HR discussion
DATA_FILE = "sampleData.json" 
DB_PATH = "chroma_db"

# -----------------------
# 2. INTELLIGENT DATA LOADER
# -----------------------
def load_medical_rag_data(filepath):
    all_docs = []
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        raw_json = json.load(f)
    
    data_content = raw_json.get("data", {})
    
    # Process OCR Data
    # sampleData.json contains 8 reports, some with identical data
    # We load them all to show the historical trend/duplicates your HR requested
    for report in data_content.get("ocrData", []):
        patient = report.get("patientName", "Mr Sukumar Jha")
        app_user_id = report.get("appUserId", "694b7fe773477b20dd627a7b")
        
        for test in report.get("diagnosticsData", []):
            # Formatted string for vector indexing
            content = (
                f"Patient: {patient}\n"
                f"- {test.get('test_name')} : {test.get('value')} {test.get('unit')}\n"
                f"ID: {app_user_id}"
            )
            all_docs.append(Document(page_content=content, metadata={"patient": patient}))

    print(f"✅ Successfully indexed {len(all_docs)} medical records.")
    return all_docs

# -----------------------
# 3. GLOBAL INITIALIZATION
# -----------------------
all_docs = load_medical_rag_data(DATA_FILE)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_KEY,
    openai_api_version=AZURE_VERSION,
    azure_deployment="text-embedding-ada-002"
)

# Clear existing DB to ensure fresh results
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

# -----------------------
# 4. CLINICAL INTELLIGENCE ENDPOINT
# -----------------------
@app.post("/ask")
def ask_question(req: QueryRequest):
    # Retrieve relevant records (increased k to get duplicates/history)
    results = db.similarity_search(req.query, k=10)
    context = "\n\n".join([r.page_content for r in results])

    # The Intelligence Prompt designed to match your example format exactly
    prompt = f"""
    You are a medical intelligence assistant.
    User Query: {req.query}
    
    Medical Context:
    {context}
    
    STRICT OUTPUT FORMAT RULES:
    1. Start with 'Patient: [Name]'.
    2. List tests using the format: - [Test Name] : [Value] [Unit]
    3. Below critical tests, add an interpretation line starting with '  → '.
    4. For Glucose Fasting 126 mg/dL, use interpretation: "This value is at the diagnostic threshold for diabetes."
    5. List multiple entries if they appear in the context to show the full record set.
    6. Summary must be a condensed version of the same bulleted format.
    7. NO markdown formatting (no bold, no italics, no headers).
    """

    response = llm.invoke(prompt)
    raw_answer = response.content.strip()
    
    # Generate the condensed summary based on the answer
    summary_prompt = f"Condense this medical response into the 3 most important points using the EXACT SAME format:\n{raw_answer}"
    summary_resp = llm.invoke(summary_prompt)
    
    return {
        "query": req.query,
        "answer": raw_answer,
        "summary": summary_resp.content.strip()
    }