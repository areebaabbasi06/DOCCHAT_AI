import sys
import os

# Add project root path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)


from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION
)

from document_loader import load_pdf
from text_splitter import split_text
from embedder import EmbeddingGenerator

import uuid



class VectorStore:


    def __init__(self):

        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

        self.embedder = EmbeddingGenerator()



    def store_documents(self, file_path):

        print("📄 Loading document...")

        text = load_pdf(file_path)


        print("✂️ Splitting text...")

        chunks = split_text(text)


        print(f"Total chunks created: {len(chunks)}")


        print("🔢 Creating embeddings...")

        embeddings = self.embedder.generate_embeddings(
            chunks
        )


        points = []


        for chunk, vector in zip(chunks, embeddings):

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),

                    vector=vector,

                    payload={
                        "text": chunk
                    }
                )
            )


        print("📤 Uploading vectors to Qdrant...")


        self.client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )


        print("✅ Documents stored successfully in Qdrant")




if __name__ == "__main__":


    vector_store = VectorStore()


    vector_store.store_documents(
        "data/sample.pdf"
    )