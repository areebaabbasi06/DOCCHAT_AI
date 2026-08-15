"""
Embedding Generator for DocChat AI
Railway-friendly FastEmbed embedding model
"""

from fastembed import TextEmbedding
from typing import List, Dict


class EmbeddingGenerator:

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5"
    ):
        print("Loading embedding model...")

        self.model = TextEmbedding(
            model_name=model_name
        )

        # BAAI/bge-small-en-v1.5 = 384 dimensions
        self.dimension = 384

        print(
            f"Model loaded successfully! "
            f"Embedding dimension: {self.dimension}"
        )


    def generate_embeddings(
        self,
        texts: List[str]
    ) -> List[List[float]]:

        if not texts:
            return []

        embeddings = list(
            self.model.embed(texts)
        )

        return [
            embedding.tolist()
            for embedding in embeddings
        ]


    def embed_chunks(
        self,
        chunks: List[Dict]
    ) -> List[Dict]:

        texts = [
            chunk["content"]
            for chunk in chunks
        ]

        embeddings = self.generate_embeddings(
            texts
        )

        for chunk, embedding in zip(
            chunks,
            embeddings
        ):
            chunk["embedding"] = embedding

        return chunks


    def embed_query(
        self,
        query: str
    ) -> List[float]:

        embedding = list(
            self.model.embed([query])
        )[0]

        return embedding.tolist()


if __name__ == "__main__":

    embedder = EmbeddingGenerator()

    test_text = [
        "Retrieval Augmented Generation is used for document question answering."
    ]

    result = embedder.generate_embeddings(
        test_text
    )

    print(
        "\n✅ Embedding Generated Successfully"
    )

    print(
        "Vector size:",
        len(result[0])
    )