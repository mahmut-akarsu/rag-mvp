from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import add_document, answer_query, load_documents

add_document("Python, backend geliştirmede çok yaygın kullanılan bir programlama dilidir.", source="seed.txt")
add_document("RAG sistemleri, arama + LLM birleşiminden oluşur.", source="seed.txt")
add_document("Google Gemini, LLM ve embedding için kullanılabilir.", source="seed.txt")

app = FastAPI(title="RAG API")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "RAG API çalışıyor! /docs adresinden API dokümantasyonuna ulaşabilirsiniz."}

@app.post("/load")
def load_endpoint():
    load_documents("data")
    return {"status": "Belgeler başarıyla yüklendi."}

@app.post("/query")
def query_response(request: QueryRequest):
    result = answer_query(request.question)
    return result
