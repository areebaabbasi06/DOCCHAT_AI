import sys
import os
from typing import Optional


# ==========================================
# PROJECT ROOT
# ==========================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RAG_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if RAG_FOLDER not in sys.path:
    sys.path.insert(0, RAG_FOLDER)


# ==========================================
# IMPORTS
# ==========================================

from embedder import EmbeddingGenerator
from retriever import QdrantRetriever
from llm import GeminiLLM
from web_search import WebSearch


# ==========================================
# RAG PIPELINE
# ==========================================

class RAGPipeline:

    def __init__(self):

        print("Initializing RAG Pipeline...")

        # ----------------------------------
        # Embedding Model
        # ----------------------------------

        self.embedder = EmbeddingGenerator()

        # ----------------------------------
        # Qdrant Retriever
        # ----------------------------------

        self.retriever = QdrantRetriever()

        # ----------------------------------
        # Gemini
        # ----------------------------------

        self.gemini = GeminiLLM()

        print("Gemini Client Initialized")

        # ----------------------------------
        # Tavily Web Search
        # ----------------------------------

        self.web_search = WebSearch()

        print("Tavily Web Search Initialized")

        print("RAG Pipeline Ready")


    # ==========================================
    # SMART WEB SEARCH DETECTION
    # ==========================================

    def needs_web_search(self, question: str) -> bool:
        """
        Decide whether the question likely needs
        fresh/current web information.

        Normal/general questions skip Tavily
        for faster responses.
        """

        text = question.lower().strip()

        web_keywords = [
            "latest",
            "today",
            "current",
            "recent",
            "news",
            "right now",
            "at the moment",
            "this week",
            "this month",
            "price",
            "weather",
            "score",
            "scores",
            "ranking",
            "rankings",
            "stock",
            "stocks",
            "market",
            "update",
            "updates",
            "election",
            "president",
            "prime minister",
            "ceo",
            "release",
            "released",
            "new version",
            "2026"
        ]

        return any(
            keyword in text
            for keyword in web_keywords
        )


    # ==========================================
    # MAIN ASK FUNCTION
    # ==========================================

    def ask(
        self,
        question: str,
        use_pdf: bool = False,
        image_data: Optional[str] = None
    ):

        # ======================================
        # VALIDATE QUESTION
        # ======================================

        if not question or not question.strip():
            return "Please enter a question."

        question = question.strip()

        print("\n===================================")
        print("Question:", question)
        print("PDF Mode:", use_pdf)
        print("Image Provided:", bool(image_data))
        print("===================================")


        # ==========================================
        # LANGUAGE INSTRUCTION
        # ==========================================

        language_instruction = """

IMPORTANT LANGUAGE RULE:

Answer the user in the same language and writing
style used by the user.

If the user asks in English, answer in English.

If the user asks in Roman Urdu, answer in Roman Urdu.

If the user asks in Urdu script, answer in Urdu script.

Do not translate the user's question into another
language unless explicitly requested.

Keep the answer clear, natural, and easy to understand.
"""


        # ==========================================
        # PDF CONTEXT
        # ==========================================

        pdf_context = ""

        if use_pdf:

            try:

                print(
                    "\nCreating query embedding..."
                )

                query_embedding = (
                    self.embedder.embed_query(
                        question
                    )
                )

                print(
                    "Searching Qdrant for PDF content..."
                )

                # Reduced from 5 to 3
                # to reduce retrieval/context time.
                results = self.retriever.search(
                    query_embedding=query_embedding,
                    top_k=3
                )

                if results:

                    pdf_parts = []

                    for item in results:

                        text = item.get(
                            "text",
                            ""
                        )

                        if text and text.strip():

                            pdf_parts.append(
                                text.strip()
                            )

                    pdf_context = "\n\n".join(
                        pdf_parts
                    )

                    # Keep PDF context smaller
                    # for faster Gemini processing.
                    if len(pdf_context) > 9000:

                        pdf_context = (
                            pdf_context[:9000]
                            + "\n[PDF context truncated]"
                        )

                if pdf_context:

                    print(
                        "PDF context found:",
                        len(pdf_context),
                        "characters"
                    )

                else:

                    print(
                        "No relevant PDF context found."
                    )

            except Exception as e:

                print(
                    "\nPDF retrieval error:"
                )

                print(
                    str(e)
                )

                pdf_context = ""

        else:

            print(
                "\nPDF not required."
            )

            print(
                "Skipping Qdrant PDF retrieval."
            )


        # ==========================================
        # IMAGE STATUS
        # ==========================================

        if image_data:

            print(
                "\nImage provided."
            )

            print(
                "Gemini will analyze the image."
            )

        else:

            print(
                "\nNo image provided."
            )


        # ==========================================
        # WEB SEARCH
        # ==========================================

        web_context = ""

        should_search_web = False


        # ==========================================
        # WEB SEARCH DECISION
        # ==========================================

        if image_data:

            # Do not automatically search the web
            # just because an image was provided.
            should_search_web = False

        elif use_pdf and pdf_context:

            # Relevant PDF information exists.
            # Use the PDF first.
            should_search_web = False

        elif use_pdf and not pdf_context:

            # PDF selected but no relevant content
            # was found. Use web as fallback.
            should_search_web = True

        else:

            # Normal chat:
            # Search web only when question appears
            # to require fresh/current information.
            should_search_web = (
                self.needs_web_search(
                    question
                )
            )


        # ==========================================
        # RUN TAVILY ONLY WHEN NEEDED
        # ==========================================

        if should_search_web:

            try:

                print(
                    "\n🌐 Searching the web with Tavily..."
                )

                web_results = (
                    self.web_search.search(
                        question,
                        max_results=2
                    )
                )

                if web_results:

                    web_parts = []

                    for item in web_results:

                        title = item.get(
                            "title",
                            ""
                        )

                        url = item.get(
                            "url",
                            ""
                        )

                        content = item.get(
                            "content",
                            ""
                        )

                        # Keep individual result small
                        if len(content) > 1800:

                            content = (
                                content[:1800]
                                + "..."
                            )

                        web_parts.append(
                            f"Title: {title}\n"
                            f"URL: {url}\n"
                            f"Content: {content}"
                        )

                    web_context = "\n\n".join(
                        web_parts
                    )

                    # Keep total web context small
                    if len(web_context) > 5000:

                        web_context = (
                            web_context[:5000]
                            + "\n[Web context truncated]"
                        )

                    print(
                        "Web results found:",
                        len(web_results)
                    )

                else:

                    print(
                        "No web results found."
                    )

            except Exception as e:

                print(
                    "\nWeb search error:"
                )

                print(
                    str(e)
                )

                web_context = ""

        else:

            print(
                "\n⚡ Tavily skipped."
            )

            print(
                "Using direct/local knowledge."
            )


        # ==========================================
        # BUILD COMBINED CONTEXT
        # ==========================================

        combined_context = ""


        # ------------------------------------------
        # LANGUAGE
        # ------------------------------------------

        combined_context += (
            "==============================\n"
            "ANSWER LANGUAGE INSTRUCTION\n"
            "==============================\n"
            + language_instruction
            + "\n\n"
        )


        # ------------------------------------------
        # PDF CONTEXT
        # ------------------------------------------

        if pdf_context:

            combined_context += (
                "==============================\n"
                "PDF DOCUMENT CONTEXT\n"
                "==============================\n\n"
                + pdf_context
                + "\n\n"
            )


        # ------------------------------------------
        # WEB CONTEXT
        # ------------------------------------------

        if web_context:

            combined_context += (
                "==============================\n"
                "WEB SEARCH CONTEXT\n"
                "==============================\n\n"
                + web_context
                + "\n\n"
            )


        # ------------------------------------------
        # IMAGE CONTEXT
        # ------------------------------------------

        if image_data:

            combined_context += (
                "==============================\n"
                "USER IMAGE\n"
                "==============================\n\n"
                "An image has been provided by the user. "
                "The image will be sent directly to Gemini "
                "for visual analysis.\n\n"
            )


        # ==========================================
        # SOURCE PRIORITY
        # ==========================================

        if pdf_context:

            combined_context += """

SOURCE PRIORITY:

The PDF document is the primary source.

Use the retrieved PDF content first.

If the PDF contains enough information,
answer from the PDF.

Do not use outside information when the
PDF already provides the answer.

"""

        elif web_context:

            combined_context += """

SOURCE PRIORITY:

Fresh web information was retrieved.

Use the web search context as the primary
external information source.

"""

        elif image_data:

            combined_context += """

SOURCE PRIORITY:

Use the user's image as the primary visual source.

Use general knowledge when appropriate.

"""

        else:

            combined_context += """

SOURCE PRIORITY:

No PDF or web context is available.

Use general knowledge when appropriate.

Do not claim that information came from a
PDF or web source when it did not.

"""


        # ==========================================
        # GENERATE ANSWER
        # ==========================================

        print(
            "\nGenerating answer with Gemini..."
        )

        try:

            result = (
                self.gemini.generate_answer(
                    question,
                    combined_context,
                    image_data=image_data
                )
            )

        except Exception as e:

            print(
                "\nGemini error:"
            )

            print(
                str(e)
            )

            return (
                "Sorry, I couldn't generate "
                "an answer right now. "
                "Please try again."
            )


        # ==========================================
        # HANDLE GEMINI RESPONSE
        # ==========================================

        if isinstance(result, dict):

            return result.get(
                "answer",
                "No answer received."
            )

        return str(result)


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    rag = RAGPipeline()

    question = (
        "What is Retrieval Augmented Generation?"
    )

    answer = rag.ask(
        question,
        use_pdf=False
    )

    print(
        "\n----- FINAL ANSWER -----\n"
    )

    print(answer)