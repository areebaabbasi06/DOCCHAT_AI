import sys
import os
import uuid

# ==========================================
# PROJECT ROOT
# ==========================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==========================================
# QDRANT IMPORTS
# ==========================================

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)


# ==========================================
# CONFIG
# ==========================================

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION
)


# ==========================================
# QDRANT DATABASE
# ==========================================

class QdrantDB:

    def __init__(self):

        print("🔌 Connecting to Qdrant Cloud...")

        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

        self.collection_name = QDRANT_COLLECTION

        print("✅ Connected to Qdrant Cloud")


    # ======================================
    # CREATE COLLECTION
    # ======================================

    def create_collection(self):

        collections = (
            self.client
            .get_collections()
            .collections
        )

        collection_names = [
            collection.name
            for collection in collections
        ]

        if self.collection_name not in collection_names:

            self.client.create_collection(
                collection_name=self.collection_name,

                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )

            print(
                f"✅ Collection "
                f"'{self.collection_name}' "
                f"created successfully."
            )

        else:

            print(
                f"✅ Collection "
                f"'{self.collection_name}' "
                f"already exists."
            )


    # ======================================
    # INSERT / UPSERT CHUNKS
    # ======================================

    def upsert_chunks(
        self,
        chunks,
        embeddings
    ):

        """
        Store document chunks and their embeddings
        inside Qdrant.

        chunks:
            List of dictionaries created by PDFProcessor.

        embeddings:
            List of 384-dimensional vectors.
        """

        # Make sure collection exists
        self.create_collection()

        if not chunks:

            print("⚠️ No chunks to insert.")

            return 0

        if not embeddings:

            print("⚠️ No embeddings to insert.")

            return 0

        if len(chunks) != len(embeddings):

            raise ValueError(
                "Number of chunks and embeddings "
                "must be the same."
            )


        points = []


        # ==================================
        # CREATE QDRANT POINTS
        # ==================================

        for chunk, embedding in zip(
            chunks,
            embeddings
        ):

            metadata = chunk.get(
                "metadata",
                {}
            )

            point = PointStruct(

                id=str(
                    uuid.uuid4()
                ),

                vector=embedding,

                payload={

                    # IMPORTANT:
                    # Retriever searches this field
                    "text": chunk.get(
                        "content",
                        ""
                    ),

                    "filename": metadata.get(
                        "filename",
                        ""
                    ),

                    "page": metadata.get(
                        "page",
                        0
                    ),

                    "chunk_index": metadata.get(
                        "chunk_index",
                        0
                    )
                }
            )

            points.append(point)


        # ==================================
        # INSERT INTO QDRANT
        # ==================================

        self.client.upsert(

            collection_name=self.collection_name,

            points=points
        )


        print(
            f"✅ {len(points)} chunks "
            f"inserted into Qdrant."
        )


        return len(points)


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    qdrant = QdrantDB()

    qdrant.create_collection()

    print("Qdrant DB ready ✅")