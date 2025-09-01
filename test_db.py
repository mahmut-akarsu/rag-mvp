
import os
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")  # ör: postgres://user:pass@localhost:5432/dbname

# Engine ve session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Test tablosu
class TestDocument(Base):
    __tablename__ = "test_documents"
    id = Column(Integer, primary_key=True)
    content = Column(Text)

# Tabloyu oluştur
Base.metadata.create_all(bind=engine)

# Test insert ve select
def test_connection():
    session = SessionLocal()
    try:
        # Test verisi ekle
        doc = TestDocument(content="DB bağlantısı başarılı mı?")
        session.add(doc)
        session.commit()
        print("[INFO] Test veri eklendi.")

        # Test verisini oku
        result = session.query(TestDocument).all()
        for r in result:
            print(f"[INFO] ID: {r.id} | Content: {r.content}")

    except Exception as e:
        print("[ERROR] DB bağlantısı başarısız:", e)
    finally:
        session.close()

if __name__ == "__main__":
    test_connection()
