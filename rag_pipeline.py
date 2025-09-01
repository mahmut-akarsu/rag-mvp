import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# .env dosyasını yükle
load_dotenv()

# Ortam değişkenlerini al
DATABASE_URL = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not DATABASE_URL or not GOOGLE_API_KEY:
    raise ValueError("DATABASE_URL and GOOGLE_API_KEY must be set in the .env file")

# PostgreSQL veritabanı bağlantısı
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class RAGPipeline:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
        self.vectorstore = None
        self._initialize_db()

    def _initialize_db(self):
        """Veritabanı tablolarını oluşturur."""
        session = SessionLocal()
        try:
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding VECTOR(768), -- Google Embeddings genellikle 768 boyutludur
                    source TEXT,
                    page_number INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            session.commit()
            print("PostgreSQL 'documents' tablosu kontrol edildi/oluşturuldu.")
        except Exception as e:
            session.rollback()
            print(f"Veritabanı başlatılırken hata oluştu: {e}")
        finally:
            session.close()

    def _clean_text(self, text: str) -> str:
        """Metinleri temizler (ekstra boşluklar, yeni satırlar vb.)."""
        text = re.sub(r'\s+', ' ', text)  # Birden fazla boşluğu tek boşluğa çevir
        text = text.strip()
        return text

    def load_and_process_pdfs(self):
        """
        Data klasöründeki PDF'leri yükler, parçalar, gömer ve PostgreSQL'e kaydeder.
        Ayrıca, FAISS vektör mağazasını oluşturur/günceller.
        """
        print(f"'{self.data_dir}' klasöründeki PDF'ler yükleniyor...")
        loader = PyPDFDirectoryLoader(self.data_dir)
        documents = loader.load()

        if not documents:
            print("Data klasöründe hiç PDF bulunamadı veya yüklenemedi.")
            return

        print(f"{len(documents)} adet PDF belgesi yüklendi.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Belgeler {len(chunks)} parçaya bölündü.")

        session = SessionLocal()
        try:
            for i, chunk in enumerate(chunks):
                content = self._clean_text(chunk.page_content)
                source = chunk.metadata.get("source")
                page_number = chunk.metadata.get("page")

                # Eğer daha önce kaydedilmişse atla veya güncelle (basit bir yaklaşım)
                # Daha gelişmiş bir sistemde değişiklikleri kontrol edebilirsiniz.
                result = session.execute(
                    text("SELECT id FROM documents WHERE content = :content AND source = :source AND page_number = :page_number"),
                    {"content": content, "source": source, "page_number": page_number}
                ).fetchone()

                if result:
                    # print(f"Parça zaten veritabanında mevcut, atlanıyor. (ID: {result[0]})")
                    continue

                # Embedding oluştur
                embedding = self.embeddings.embed_query(content)

                # PostgreSQL'e kaydet
                session.execute(
                    text("""
                        INSERT INTO documents (content, embedding, source, page_number)
                        VALUES (:content, :embedding, :source, :page_number)
                    """),
                    {
                        "content": content,
                        "embedding": embedding,
                        "source": source,
                        "page_number": page_number,
                    }
                )
                if (i + 1) % 100 == 0:
                    session.commit()
                    print(f"{i + 1} parça işlendi ve kaydedildi...")

            session.commit()
            print("Tüm PDF parçaları veritabanına başarıyla kaydedildi.")

            # FAISS vektör mağazasını veritabanındaki verilerle oluştur/güncelle
            self._build_faiss_vectorstore_from_db()

        except Exception as e:
            session.rollback()
            print(f"PDF işlenirken veya veritabanına kaydedilirken hata oluştu: {e}")
        finally:
            session.close()

    def _build_faiss_vectorstore_from_db(self):
        """PostgreSQL'den metinleri ve embeddingleri çekerek FAISS vektör mağazasını oluşturur."""
        print("Veritabanından veriler çekilerek FAISS vektör mağazası oluşturuluyor...")
        session = SessionLocal()
        try:
            results = session.execute(text("SELECT content, embedding FROM documents")).fetchall()
            
            if not results:
                print("Veritabanında hiç belge bulunamadı. FAISS oluşturulamadı.")
                self.vectorstore = None
                return

            texts = [r[0] for r in results]
            embeddings_list = [r[1] for r in results]

            # FAISS'in doğrudan embedding listesinden oluşturulması
            # Langchain'in FAISS.from_texts veya FAISS.from_documents metotları genellikle embeddingleri kendisi hesaplar.
            # Burada zaten embeddinglerimiz olduğu için biraz daha manuel bir yol izleyebiliriz veya
            # geçici olarak dummy Document nesneleri oluşturup from_documents kullanabiliriz.
            # Basitlik adına, burada Langchain'in embedding oluşturma özelliğini kullanmak için metinleri veriyoruz.
            # Ancak FAISS.from_embeddings diye bir metod yoktur.
            # Bu yüzden, ya embeddingleri ve metinleri ayrı ayrı FAISS Index'e ekleriz (daha karmaşık)
            # ya da burada `FAISS.from_documents`'ı kullanarak embeddingleri tekrar hesaplatırız.
            # İlk yüklemede embedding'leri zaten hesapladığımız için,
            # FAISS'i doğrudan oluşturmak yerine, metinlerle birlikte embeddingleri kullanarak bir FAISS Index'i oluştururuz.
            
            # Langchain'in FAISS entegrasyonu genellikle kendi embedding'lerini oluşturmayı bekler.
            # Veritabanındaki embedding'leri kullanmak için farklı bir yaklaşım izlememiz gerekiyor.
            # Şimdilik, veritabanındaki metinleri alıp FAISS'in tekrar embedding oluşturmasına izin verelim,
            # bu da sistemin daha az karmaşık olmasını sağlar, ancak performans kaybı olabilir.
            # Daha optimize bir çözüm için, FAISS index'ini doğrudan embedding'lerden oluşturup
            # metinleri metadata olarak eklemek gerekir.

            # Bu örnek için, basitçe veritabanındaki metinleri kullanarak FAISS'i yeniden oluşturalım:
            from langchain.schema import Document
            db_documents = [Document(page_content=text) for text in texts]
            self.vectorstore = FAISS.from_documents(db_documents, self.embeddings)
            
            print("FAISS vektör mağazası veritabanı verileriyle başarıyla oluşturuldu.")
        except Exception as e:
            print(f"FAISS vektör mağazası oluşturulurken hata oluştu: {e}")
            self.vectorstore = None
        finally:
            session.close()

    def get_answer(self, question: str) -> str:
        """Kullanıcının sorusuna RAG kullanarak cevap verir."""
        if not self.vectorstore:
            return "Vektör mağazası hazır değil. Lütfen önce PDF'leri yükleyin ve işleyin."

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

# İlk başlatmada PDF'leri yüklemek için bir örnek
if __name__ == "__main__":
    rag_pipeline = RAGPipeline()
    rag_pipeline.load_and_process_pdfs() # Bu fonksiyon veritabanına kaydeder ve FAISS'i günceller
    
    if rag_pipeline.vectorstore:
        while True:
            query = input("\nSorunuz (çıkmak için 'q'): ")
            if query.lower() == 'q':
                break
            answer = rag_pipeline.get_answer(query)
            print(f"\nCevap: {answer}")
    else:
        print("FAISS vektör mağazası oluşturulamadığı için soru sorulamıyor.")