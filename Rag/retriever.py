import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from typing import List, Dict

from qdrant_client import QdrantClient

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION
)



class QdrantRetriever:


    def __init__(self):

        self.collection_name = QDRANT_COLLECTION

        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

        print("✅ Connected to Qdrant Cloud")



    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict]:


        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k
        )


        chunks = []


        for result in results.points:

            chunks.append(
                {
                    "text": result.payload.get("text"),
                    "score": result.score
                }
            )


        return chunks



if __name__ == "__main__":


    retriever = QdrantRetriever()

    print("Retriever ready ✅")