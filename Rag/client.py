import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from config import GEMINI_API_KEY
from google import genai


class GeminiClient:

    def __init__(self):

        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not found in .env file."
            )

        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        self.model = "gemini-3.5-flash"

        print("✅ Gemini Client Initialized")


    def generate_answer(
        self,
        question: str,
        context: str
    ):

        prompt = f"""
You are a helpful AI assistant for a RAG system.

Answer the user's question using the provided PDF
and web context.

Rules:

1. Answer in the same language as the user's question.
2. Use the provided context.
3. Do not invent information.
4. If the answer is not available in the context,
   clearly say that you don't know.
5. Give a clear and concise answer.

Context:

{context}

Question:

{question}

Answer:
"""

        try:

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            if not response.text:

                return {
                    "answer": "Gemini returned an empty response.",
                    "sources": []
                }

            return {
                "answer": response.text.strip(),
                "sources": []
            }

        except Exception as e:

            return {
                "answer": f"An error occurred: {str(e)}",
                "sources": []
            }


if __name__ == "__main__":

    gemini = GeminiClient()

    answer = gemini.generate_answer(
        "What is RAG?",
        "RAG combines LLMs with external knowledge sources."
    )

    print("\nAnswer:")
    print(answer)