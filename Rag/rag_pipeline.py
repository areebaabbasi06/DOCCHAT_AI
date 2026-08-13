import sys
import os

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

        print("✅ Gemini Client Initialized")

        # ----------------------------------
        # Tavily Web Search
        # ----------------------------------

        self.web_search = WebSearch()

        print("✅ Tavily Web Search Initialized")

        print("✅ RAG Pipeline Ready")


    # ==========================================
    # MAIN ASK FUNCTION
    # ==========================================

    def ask(
        self,
        question: str,
        use_pdf: bool = True
    ):

        """
        Three modes are supported:

        1. PDF + Internet
           use_pdf=True
           → Search PDF in Qdrant
           → Search Internet with Tavily
           → Gemini combines both

        2. Internet only
           use_pdf=False
           → Skip PDF
           → Search Internet
           → Gemini answers from web context

        3. General chat
           use_pdf=False
           → If web has no useful result,
             Gemini can answer normally.
        """

        # ==================================
        # VALIDATE QUESTION
        # ==================================

        if not question or not question.strip():

            return "Please enter a question."

        question = question.strip()

        print("\n===================================")
        print("Question:", question)
        print("PDF Mode:", use_pdf)
        print("===================================")


        # ==================================
        # PDF CONTEXT
        # ==================================

        pdf_context = ""

        if use_pdf:

            try:

                print("\n🔍 Creating query embedding...")

                query_embedding = (
                    self.embedder.embed_query(
                        question
                    )
                )

                print("🔎 Searching Qdrant...")

                # Reduced from 5 to 3
                # This improves speed and keeps
                # the prompt smaller.

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

                    # Keep only useful amount
                    # of PDF context.

                    pdf_context = "\n\n".join(
                        pdf_parts
                    )

                    # Limit context size
                    if len(pdf_context) > 12000:

                        pdf_context = (
                            pdf_context[:12000]
                            + "\n[PDF context truncated]"
                        )

                if pdf_context:

                    print(
                        "✅ PDF context found "
                        f"({len(pdf_context)} characters)"
                    )

                else:

                    print(
                        "ℹ️ No relevant PDF context found."
                    )

            except Exception as e:

                print(
                    "\n⚠️ PDF retrieval error:"
                )

                print(str(e))

                # PDF error should NOT stop
                # internet chat.

                pdf_context = ""

        else:

            print(
                "\n📄 PDF not required."
            )

            print(
                "➡️ Skipping Qdrant/PDF retrieval."
            )


        # ==========================================
        # WEB SEARCH
        # ==========================================

        web_context = ""

        try:

            print("\n🌐 Searching the web...")

            # Reduced from 5 to 3 results
            # to improve response speed.

            web_results = self.web_search.search(
                question,
                max_results=3
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

                    # Limit each web result
                    # so Gemini gets less text.

                    if len(content) > 3000:

                        content = (
                            content[:3000]
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

                # Overall web context limit

                if len(web_context) > 9000:

                    web_context = (
                        web_context[:9000]
                        + "\n[Web context truncated]"
                    )

                print(
                    "✅ Web results found: "
                    f"{len(web_results)}"
                )

            else:

                print(
                    "ℹ️ No web results found."
                )

        except Exception as e:

            print(
                "\n⚠️ Web search error:"
            )

            print(str(e))

            web_context = ""


        # ==========================================
        # BUILD COMBINED CONTEXT
        # ==========================================

        combined_context = ""


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
        # NO EXTERNAL CONTEXT
        # ------------------------------------------

        if not combined_context:

            combined_context = (

                "No relevant PDF or web context "
                "was available. You may answer "
                "using your general knowledge, "
                "but do not pretend that the "
                "answer came from a PDF or web source."
            )


        # ==========================================
        # GENERATE ANSWER
        # ==========================================

        print(
            "\n🤖 Generating answer with Gemini..."
        )

        try:

            result = (
                self.gemini.generate_answer(
                    question,
                    combined_context
                )
            )

        except Exception as e:

            print(
                "\n❌ Gemini error:"
            )

            print(str(e))

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

    # --------------------------------------
    # Test WITHOUT PDF
    # --------------------------------------

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