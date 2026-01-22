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

    # HR's System Prompt Integration
    system_prompt = f"""
    You are an AI assistant operating within a Retrieval-Augmented Generation (RAG) system that processes sensitive patient medical information.
    RETRIEVED DATA:
    {context}

    YOUR TASKS:
    1. Use only the retrieved patient-specific data provided in the context.
    2. Interpret information accurately and conservatively. Generate clear, contextual, and relevant answers.
    3. Avoid speculation and do not introduce information not present in the data.
    4. Clearly indicate uncertainty or missing information when data is insufficient (e.g., specific dates like 18th of June).
    5. You can provide diagnoses or medical advice if supported by data, but ALWAYS mention: "This information is not medical advice. Please consult a doctor for actual medical advice."

    OUTPUT FORMAT:
    - [Test Name] : [Value] [Unit]
      → [Clinical Interpretation/Trend]
    
    Summary:
    [Bullet points highlighting trends or critical actions]
    """

    response = llm.invoke(system_prompt + f"\n\nUser Query: {req.query}")
    raw_answer = response.content.strip()
    
    # summary generation following the same bullet format
    summary_prompt = f"Condense this medical analysis into 3 key bullet points using the approved format:\n{raw_answer}"
    summary_resp = llm.invoke(summary_prompt)
    
    return {
        "query": req.query,
        "answer": raw_answer,
        "summary": summary_resp.content.strip()
    }