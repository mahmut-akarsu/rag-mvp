import os, glob
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

EMBED_MODEL = "models/embedding-001"
DIMENSION = 768  # embedding-001 çıktısı

index = faiss.IndexFlatL2(DIMENSION)
documents: list[str] = []  
metadatas: list[dict] = []  

def embed_text(text: str) -> np.ndarray:
    r = genai.embed_content(model=EMBED_MODEL, content=text)
    return np.array(r["embedding"], dtype="float32")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """Sabit uzunlukta karakter bazlı chunk'lar üretir."""
    chunks = []
    start = 0
    cid = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append((text[start:end], cid))
        start = end - overlap
        cid += 1
    return chunks

def add_document(doc_text: str, source: str = "unknown", page: int | None = None, chunk_id: int | None = None):
    vec = embed_text(doc_text)
    index.add(np.array([vec]))
    documents.append(doc_text)
    metadatas.append({"source": source, "page": page, "chunk_id": chunk_id})

def load_documents(folder: str = "data"):
    files = glob.glob(f"{folder}/*")
    for file in files:
        ext = file.split(".")[-1].lower()
        basename = os.path.basename(file)

        if ext == "pdf":
            reader = PdfReader(file)
            for page_no, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue
                for chunk_text_val, cid in chunk_text(page_text, chunk_size=1000, overlap=200):
                    add_document(chunk_text_val, source=basename, page=page_no, chunk_id=cid)

        elif ext == "txt":
            with open(file, "r", encoding="utf-8") as f:
                txt = f.read()
            for chunk_text_val, cid in chunk_text(txt, chunk_size=1000, overlap=200):
                add_document(chunk_text_val, source=basename, page=None, chunk_id=cid)

        else:
            continue

def search(query: str, top_k: int = 3):
    qv = embed_text(query)
    D, I = index.search(np.array([qv]), top_k)
    results = []
    for rank, i in enumerate(I[0]):
        if i == -1:
            continue
        results.append({
            "text": documents[i],
            "meta": metadatas[i],          
            "score": float(D[0][rank])     
        })
    return results

def answer_query(query: str, top_k: int = 3, model_name: str = "models/gemini-2.0-flash"):
    results = search(query, top_k=top_k)

    print("\n[DEBUG] Context için seçilen dokümanlar:")
    for r in results:
        src = r["meta"].get("source")
        pg  = r["meta"].get("page")
        print(f"- {src} s.{pg}: {r['text'][:100]}...")

    stitched = []
    for r in results:
        src = r["meta"].get("source")
        pg  = r["meta"].get("page")
        stitched.append(f"[Kaynak: {src} s.{pg}]\n{r['text']}")
    context = "\n\n---\n\n".join(stitched)

    prompt = (
        "Aşağıdaki bağlam parçalarına dayanarak soruyu yanıtla. "
        "Varsayım yapma; emin değilsen bunu söyle. Yanıtı Türkçe, kısa ve net ver.\n\n"
        f"Soru: {query}\n\nBağlam:\n{context}\n\nCevap:"
    )

    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    answer = resp.text


    sources = []
    for r in results:
        md = r["meta"]
        sources.append({
            "doc": md.get("source"),
            "page": md.get("page"),
            "chunk_id": md.get("chunk_id"),
            "excerpt": r["text"][:200]
        })

    return {"answer": answer, "sources": sources}
