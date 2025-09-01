import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

# .env dosyasını yükle
load_dotenv()

# Ortam değişkenlerini al
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")  # varsayılan local
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY must be set in the .env file")

class RAGPipeline:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
        self.vectorstore = None
        self.qdrant_client = QdrantClient(url=QDRANT_URL)
        print(f"Qdrant'a bağlandı: {QDRANT_URL}")

    def _clean_text(self, text: str) -> str:
        """Metinleri temizler (ekstra boşluklar, yeni satırlar vb.)."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def load_and_process_pdfs(self):
        """PDF'leri yükle, parçalara ayır, Qdrant'a kaydet."""
        print(f"'{self.data_dir}' klasöründeki PDF'ler yükleniyor...")
        loader = PyPDFDirectoryLoader(self.data_dir)
        documents = loader.load()

        if not documents:
            print("Data klasöründe hiç PDF bulunamadı.")
            return

        print(f"{len(documents)} adet PDF belgesi yüklendi.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Belgeler {len(chunks)} parçaya bölündü.")

        # Qdrant koleksiyonunu oluştur (eğer yoksa)
        self.qdrant_client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={"size": 768, "distance": "Cosine"}  # Google Embedding boyutu 768
        )

        # Qdrant vectorstore oluştur
        self.vectorstore = Qdrant.from_documents(
            chunks,
            self.embeddings,
            url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION,
        )

        print("PDF parçaları Qdrant'a başarıyla kaydedildi.")

    def get_answer(self, question: str) -> str:
        """Kullanıcının sorusuna RAG kullanarak cevap verir."""
        if not self.vectorstore:
            return "Qdrant vectorstore hazır değil. Lütfen önce PDF'leri yükleyin."

        print(f"Soruya cevap aranıyor: '{question}'")
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True
        )
        response = qa_chain({"query": question})

        answer = response["result"]
        source_documents = response["source_documents"]

        sources_info = []
        for doc in source_documents:
            source = doc.metadata.get("source", "Bilinmeyen Kaynak")
            page = doc.metadata.get("page", "Bilinmeyen Sayfa")
            sources_info.append(f"Kaynak: {os.path.basename(source)}, Sayfa: {page}")

        if sources_info:
            answer += "\n\nKaynaklar:\n" + "\n".join(sources_info)

        return answer

# Çalıştırma
if __name__ == "__main__":
    rag_pipeline = RAGPipeline()
    rag_pipeline.load_and_process_pdfs()
    
    if rag_pipeline.vectorstore:
        while True:
            query = input("\nSorunuz (çıkmak için 'q'): ")
            if query.lower() == 'q':
                break
            answer = rag_pipeline.get_answer(query)
            print(f"\nCevap: {answer}")
    else:
        print("Qdrant vectorstore oluşturulamadı.")
