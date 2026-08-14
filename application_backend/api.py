from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import os
import shutil
import sys
import threading


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
# APPLICATION BACKEND IMPORTS
# ==========================================

from application_backend.file_validator import validate_file
from application_backend.session_manager import SessionManager


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="DocChat AI Backend"
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# SHARED SERVICES
# ==========================================

session_manager = SessionManager()


# ==========================================
# LAZY SERVICES
# ==========================================

pdf_processor = None
qdrant_db = None
rag = None

rag_lock = threading.Lock()


def get_services():
    """
    Load PDF processor and Qdrant only when needed.
    """

    global pdf_processor
    global qdrant_db

    if pdf_processor is None:
        from Rag.pdf_processor import PDFProcessor
        pdf_processor = PDFProcessor()

    if qdrant_db is None:
        from Rag.qdrant_db import QdrantDB
        qdrant_db = QdrantDB()

    return pdf_processor, qdrant_db


def get_rag():
    """
    Load RAG pipeline only when PDF/chat actually needs it.

    This prevents the embedding model from loading
    during Railway container startup.
    """

    global rag

    if rag is None:

        with rag_lock:

            if rag is None:

                print(
                    "\n🧠 Loading RAG Pipeline..."
                )

                from Rag.rag_pipeline import RAGPipeline

                rag = RAGPipeline()

                print(
                    "✅ RAG Pipeline loaded successfully."
                )

    return rag


# ==========================================
# UPLOAD FOLDER
# ==========================================

UPLOAD_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ==========================================
# HEALTH CHECK / HOME
# ==========================================

@app.get("/")
def home():

    return {
        "status": "ok",
        "message": "DocChat AI Backend Running"
    }


# ==========================================
# PDF UPLOAD
# ==========================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    try:

        # ==================================
        # SAVE PDF
        # ==================================

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        print(
            f"\n📄 PDF uploaded: {file.filename}"
        )


        # ==================================
        # VALIDATE PDF
        # ==================================

        valid, message = validate_file(
            file_path
        )

        if not valid:

            if os.path.exists(file_path):
                os.remove(file_path)

            return {
                "status": "error",
                "message": message,
                "pages": 0,
                "chunks": 0
            }


        print(
            "✅ PDF validation successful"
        )


        # ==================================
        # LOAD SERVICES
        # ==================================

        print(
            "🔧 Loading PDF/Qdrant services..."
        )

        pdf_processor_service, qdrant_service = (
            get_services()
        )


        # ==================================
        # EXTRACT TEXT + CREATE CHUNKS
        # ==================================

        print(
            "📖 Extracting text from PDF..."
        )

        chunks = pdf_processor_service.process_pdf(
            file_path,
            filename=file.filename
        )


        # ==================================
        # CHECK CHUNKS
        # ==================================

        if not chunks:

            return {
                "status": "error",
                "message": (
                    "No text/chunks could be "
                    "extracted from the PDF."
                ),
                "pages": 0,
                "chunks": 0
            }


        # ==================================
        # CALCULATE PAGE COUNT
        # ==================================

        pages = len(
            set(
                chunk["metadata"]["page"]
                for chunk in chunks
            )
        )


        print(
            f"📄 Pages processed: {pages}"
        )

        print(
            f"✂️ Chunks created: {len(chunks)}"
        )


        # ==================================
        # LOAD RAG ONLY NOW
        # ==================================

        print(
            "🧠 Loading embedding model..."
        )

        rag_service = get_rag()


        # ==================================
        # CREATE EMBEDDINGS
        # ==================================

        print(
            "🧠 Creating embeddings..."
        )

        embeddings = []


        for index, chunk in enumerate(chunks):

            content = chunk.get(
                "content",
                ""
            )

            if not content.strip():
                continue


            embedding = (
                rag_service.embedder.embed_query(
                    content
                )
            )


            embeddings.append(
                embedding
            )


            if (
                index == 0
                or
                (index + 1) % 10 == 0
                or
                index == len(chunks) - 1
            ):

                print(
                    f"   Embedding "
                    f"{index + 1}/{len(chunks)}"
                )


        # ==================================
        # VERIFY EMBEDDINGS
        # ==================================

        if len(embeddings) != len(chunks):

            raise ValueError(
                "Number of embeddings does not "
                "match number of chunks."
            )


        print(
            f"✅ Embeddings created: "
            f"{len(embeddings)}"
        )


        # ==================================
        # STORE IN QDRANT
        # ==================================

        print(
            "📦 Storing chunks in Qdrant..."
        )


        stored_chunks = (
            qdrant_service.upsert_chunks(
                chunks,
                embeddings
            )
        )


        print(
            f"✅ Stored {stored_chunks} "
            f"chunks in Qdrant."
        )


        # ==================================
        # SUCCESS RESPONSE
        # ==================================

        return {

            "status": "success",

            "message":
                "PDF processed successfully.",

            "file":
                file.filename,

            "pages":
                pages,

            "chunks":
                stored_chunks
        }


    except Exception as e:

        print(
            "\n❌ PDF processing error:"
        )

        print(
            str(e)
        )


        return {

            "status": "error",

            "message":
                str(e),

            "pages": 0,

            "chunks": 0
        }


# ==========================================
# PDF VIEWER
# ==========================================

@app.get("/pdf/{filename}")
def get_pdf(filename: str):

    file_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    if not os.path.exists(file_path):

        return {
            "status": "error",
            "message": "PDF not found."
        }


    return FileResponse(
        file_path,
        media_type="application/pdf"
    )


# ==========================================
# CREATE NEW SESSION
# ==========================================

@app.post("/session")
def create_session():

    session_id = (
        session_manager.create_session()
    )


    return {

        "status": "success",

        "session_id":
            session_id
    }


# ==========================================
# GET ALL CHAT HISTORY
# ==========================================

@app.get("/history")
def get_history():

    sessions = (
        session_manager.get_all_sessions()
    )


    return {

        "status": "success",

        "sessions":
            sessions
    }


# ==========================================
# GET ONE OLD CHAT
# ==========================================

@app.get("/history/{session_id}")
def get_old_chat(
    session_id: str
):

    history = (
        session_manager.get_history(
            session_id
        )
    )


    return {

        "status": "success",

        "session_id":
            session_id,

        "messages":
            history
    }


# ==========================================
# DELETE CHAT HISTORY
# ==========================================

@app.delete("/history/{session_id}")
def delete_chat(
    session_id: str
):

    deleted = (
        session_manager.delete_session(
            session_id
        )
    )


    if not deleted:

        return {

            "status": "error",

            "message":
                "Session not found."
        }


    return {

        "status": "success",

        "message":
            "Chat deleted successfully."
    }


# ==========================================
# CHAT ENDPOINT
# ==========================================

@app.post("/chat")
async def chat(
    request: dict
):

    # ======================================
    # GET QUESTION
    # ======================================

    question = request.get(
        "question",
        ""
    )


    # ======================================
    # GET SESSION
    # ======================================

    session_id = request.get(
        "session_id"
    )


    # ======================================
    # PDF MODE
    # ======================================

    use_pdf = request.get(
        "use_pdf",
        False
    )

    use_pdf = bool(use_pdf)


    # ======================================
    # GET IMAGE
    # ======================================

    image_data = request.get(
        "image",
        None
    )


    # ======================================
    # VALIDATE IMAGE
    # ======================================

    if image_data is not None:

        if not isinstance(
            image_data,
            str
        ):

            return {

                "status": "error",

                "message":
                    "Invalid image data."
            }


        image_data = image_data.strip()


        if not image_data:
            image_data = None


        if image_data and len(image_data) > 10_000_000:

            return {

                "status": "error",

                "message":
                    "Image is too large. "
                    "Please choose a smaller image."
            }


        if image_data:

            if not image_data.startswith(
                "data:image/"
            ):

                return {

                    "status": "error",

                    "message":
                        "Invalid image format."
                }


    # ======================================
    # VALIDATE QUESTION
    # ======================================

    if not isinstance(
        question,
        str
    ):

        question = str(question)


    question = question.strip()


    if not question:

        return {

            "status": "error",

            "message":
                "Question cannot be empty."
        }


    # ======================================
    # CREATE SESSION IF NEEDED
    # ======================================

    if not session_id:

        session_id = (
            session_manager.create_session()
        )


    try:

        print(
            "\n==================================="
        )

        print(
            "💬 Question:",
            question
        )

        print(
            "📄 PDF Retrieval:",
            use_pdf
        )

        print(
            "🖼️ Image:",
            "Yes" if image_data else "No"
        )

        print(
            "==================================="
        )


        # ==================================
        # PDF STATUS
        # ==================================

        if use_pdf:

            print(
                "📄 PDF retrieval enabled."
            )

        else:

            print(
                "📄 PDF not required."
            )

            print(
                "➡️ Skipping Qdrant/PDF retrieval."
            )


        # ==================================
        # IMAGE STATUS
        # ==================================

        if image_data:

            print(
                "🖼️ Image received."
            )

            print(
                "➡️ Image will be sent to Gemini."
            )

        else:

            print(
                "🖼️ No image attached."
            )


        # ==================================
        # LOAD RAG ONLY WHEN CHAT IS USED
        # ==================================

        print(
            "\n🤖 Loading RAG pipeline..."
        )

        rag_service = get_rag()


        print(
            "🤖 Sending request to RAG pipeline..."
        )


        answer = rag_service.ask(
            question,
            use_pdf=use_pdf,
            image_data=image_data
        )


        # ==================================
        # SAVE CHAT
        # ==================================

        session_manager.add_message(
            session_id,
            question,
            answer
        )


        # ==================================
        # RESPONSE
        # ==================================

        return {

            "status": "success",

            "session_id":
                session_id,

            "question":
                question,

            "answer":
                answer
        }


    except Exception as e:

        print(
            "\n❌ Chat error:"
        )

        print(
            str(e)
        )


        return {

            "status": "error",

            "message":
                str(e)
        }