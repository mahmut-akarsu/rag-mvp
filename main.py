from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from qdrant_pipeline import RAGPipeline  # burayı rag_pipeline yerine qdrant_pipeline yaptım
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware

# FastAPI uygulamasını başlat
app = FastAPI(title="RAG PDF Chatbot API with Qdrant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend port
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, OPTIONS vs. tüm methodlar
    allow_headers=["*"],
)

# RAGPipeline örneğini global olarak oluştur
# Bu, uygulamanın her isteğinde yeniden başlatılmasını engeller
rag_pipeline_instance: Optional[RAGPipeline] = None

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    """
    Uygulama başladığında RAGPipeline'ı başlatır ve PDF'leri yükler.
    """
    global rag_pipeline_instance
    print("Uygulama başlatılıyor...")
    rag_pipeline_instance = RAGPipeline()
    
    # PDF'leri işle ve Qdrant'a kaydet
    print("PDF'ler yükleniyor ve işleniyor...")
    rag_pipeline_instance.load_and_process_pdfs()
    print("PDF işleme tamamlandı. RAG sistemi hazır.")
    if not rag_pipeline_instance.vectorstore:
        print("UYARI: Qdrant vectorstore başlangıçta oluşturulamadı. Sorulara cevap verilemeyebilir.")

@app.get("/", status_code=status.HTTP_200_OK)
async def read_root():
    """Basit bir sağlık kontrolü endpoint'i."""
    return {"message": "RAG PDF Chatbot API with Qdrant is running!"}

@app.post("/query", status_code=status.HTTP_200_OK)
async def query_rag(request: QueryRequest):
    """
    Kullanıcının sorusunu alır ve RAG sistemi aracılığıyla bir cevap döndürür.
    """
    global rag_pipeline_instance
    if rag_pipeline_instance is None or rag_pipeline_instance.vectorstore is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG sistemi henüz hazır değil veya başlatılamadı."
        )

    try:
        print(f"API isteği alındı: '{request.query}'")
        answer = rag_pipeline_instance.get_answer(request.query)
        return {"answer": answer}
    except Exception as e:
        print(f"Soru işlenirken hata oluştu: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Soru işlenirken bir hata oluştu: {e}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
