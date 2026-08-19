from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

import os
import sys
import threading
import time


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
services_lock = threading.Lock()


def get_services():
    """
    Load PDF processor and Qdrant only when needed.
    """

    global pdf_processor
    global qdrant_db

    with services_lock:

        if pdf_processor is None:

            print(
                "📖 Initializing PDF processor..."
            )

            from Rag.pdf_processor import PDFProcessor

            pdf_processor = PDFProcessor()

            print(
                "✅ PDF processor ready."
            )

        if qdrant_db is None:

            print(
                "🔌 Initializing Qdrant..."
            )

            from Rag.qdrant_db import QdrantDB

            qdrant_db = QdrantDB()

            print(
                "✅ Qdrant service ready."
            )

    return pdf_processor, qdrant_db


def get_rag():
    """
    Load RAG pipeline only when actually required.
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

    # ======================================
    # BASIC VALIDATION
    # ======================================

    if not file.filename:

        return {
            "status": "error",
            "message": "No PDF file was selected.",
            "pages": 0,
            "chunks": 0
        }

    filename = os.path.basename(
        file.filename
    )

    if not filename.lower().endswith(".pdf"):

        return {
            "status": "error",
            "message": "Please upload a PDF file.",
            "pages": 0,
            "chunks": 0
        }

    file_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    try:

        print(
            "\n==================================="
        )

        print(
            "📥 PDF UPLOAD STARTED"
        )

        print(
            "📄 Filename:",
            filename
        )

        print(
            "==================================="
        )


        # ==================================
        # SAVE PDF IN SMALL STREAMS
        # ==================================

        print(
            "💾 Saving PDF..."
        )

        with open(
            file_path,
            "wb"
        ) as buffer:

            while True:

                data = await file.read(
                    1024 * 1024
                )

                if not data:
                    break

                buffer.write(
                    data
                )

        await file.close()

        print(
            "✅ PDF saved successfully."
        )


        # ==================================
        # VALIDATE PDF
        # ==================================

        print(
            "🔍 Validating PDF..."
        )

        valid, message = await run_in_threadpool(
            validate_file,
            file_path
        )

        if not valid:

            print(
                "❌ PDF validation failed:",
                message
            )

            if os.path.exists(file_path):

                os.remove(
                    file_path
                )

            return {
                "status": "error",
                "message": message,
                "pages": 0,
                "chunks": 0
            }

        print(
            "✅ PDF validation successful."
        )


        # ==================================
        # LOAD PDF PROCESSOR
        # ==================================

        print(
            "🔧 Loading PDF processor..."
        )

        pdf_processor_service, qdrant_service = (
            await run_in_threadpool(
                get_services
            )
        )


        # ==================================
        # EXTRACT PDF TEXT + CHUNKS
        # ==================================

        print(
            "📖 Extracting PDF text..."
        )

        chunks = await run_in_threadpool(
            pdf_processor_service.process_pdf,
            file_path,
            filename=filename
        )

        print(
            "✅ PDF text extraction completed."
        )


        # ==================================
        # CHECK CHUNKS
        # ==================================

        if not chunks:

            print(
                "❌ No chunks were created."
            )

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
        # PAGE COUNT
        # ==================================

        pages = len(
            set(
                chunk.get(
                    "metadata",
                    {}
                ).get(
                    "page",
                    0
                )
                for chunk in chunks
            )
        )

        total_chunks = len(
            chunks
        )

        print(
            f"📄 Pages processed: {pages}"
        )

        print(
            f"✂️ Chunks created: {total_chunks}"
        )


        # ==================================
        # LOAD RAG / EMBEDDING MODEL
        # ==================================

        print(
            "🧠 Loading embedding model..."
        )

        rag_service = await run_in_threadpool(
            get_rag
        )

        print(
            "✅ Embedding model ready."
        )


        # ==================================
        # EMBEDDING + QDRANT BATCHING
        # ==================================

        # Smaller batch size helps reduce
        # Qdrant timeout problems.
        BATCH_SIZE = 4

        # Number of times to retry a failed
        # Qdrant upload.
        MAX_RETRIES = 3

        # Seconds between retries.
        RETRY_DELAY = 2

        total_stored = 0

        print(
            f"🧠 Processing embeddings "
            f"in batches of {BATCH_SIZE}..."
        )

        print(
            f"🔁 Qdrant max retries per batch: "
            f"{MAX_RETRIES}"
        )


        # ==================================
        # PROCESS ALL CHUNKS
        # ==================================

        for start in range(
            0,
            total_chunks,
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                total_chunks
            )

            batch_chunks = chunks[
                start:end
            ]

            print(
                f"\n🧠 Embedding chunks "
                f"{start + 1}-{end}/"
                f"{total_chunks}"
            )


            # ==================================
            # COLLECT TEXT
            # ==================================

            batch_texts = []

            for chunk in batch_chunks:

                content = chunk.get(
                    "content",
                    ""
                )

                if not content.strip():

                    raise ValueError(
                        "A PDF chunk contains "
                        "empty text."
                    )

                batch_texts.append(
                    content
                )


            # ==================================
            # CREATE BATCH EMBEDDINGS
            # ==================================

            print(
                f"🧠 Generating "
                f"{len(batch_texts)} embeddings..."
            )

            batch_embeddings = await run_in_threadpool(
                rag_service.embedder.generate_embeddings,
                batch_texts
            )

            print(
                f"✅ Created "
                f"{len(batch_embeddings)} "
                f"embeddings."
            )


            # ==================================
            # STORE BATCH IN QDRANT
            # ==================================

            print(
                "📦 Sending batch to Qdrant..."
            )

            stored = None
            last_error = None


            # ==================================
            # QDRANT RETRY LOOP
            # ==================================

            for attempt in range(
                1,
                MAX_RETRIES + 1
            ):

                try:

                    print(
                        f"🔄 Qdrant upload attempt "
                        f"{attempt}/{MAX_RETRIES}"
                    )

                    stored = await run_in_threadpool(
                        qdrant_service.upsert_chunks,
                        batch_chunks,
                        batch_embeddings
                    )

                    print(
                        "✅ Qdrant upload successful."
                    )

                    break


                except Exception as qdrant_error:

                    last_error = qdrant_error

                    print(
                        f"⚠️ Qdrant upload failed "
                        f"on attempt {attempt}: "
                        f"{type(qdrant_error).__name__}: "
                        f"{str(qdrant_error)}"
                    )


                    # ==================================
                    # RETRY
                    # ==================================

                    if attempt < MAX_RETRIES:

                        print(
                            f"⏳ Waiting "
                            f"{RETRY_DELAY} seconds "
                            f"before retry..."
                        )

                        await run_in_threadpool(
                            time.sleep,
                            RETRY_DELAY
                        )

                    else:

                        print(
                            "❌ Qdrant upload failed "
                            "after all retries."
                        )


            # ==================================
            # CHECK QDRANT RESULT
            # ==================================

            if stored is None:

                if last_error is not None:

                    raise last_error

                stored = len(
                    batch_chunks
                )


            total_stored += int(
                stored
            )

            print(
                f"✅ Batch stored. "
                f"Total stored: "
                f"{total_stored}/{total_chunks}"
            )


            # ==================================
            # RELEASE BATCH MEMORY
            # ==================================

            del batch_texts
            del batch_embeddings
            del batch_chunks


        # ==================================
        # FINAL SUCCESS
        # ==================================

        print(
            "\n==================================="
        )

        print(
            "✅ PDF PROCESSING COMPLETE"
        )

        print(
            f"📄 Pages: {pages}"
        )

        print(
            f"✂️ Chunks: {total_stored}"
        )

        print(
            "==================================="
        )


        return {

            "status": "success",

            "message":
                "PDF processed successfully.",

            "file":
                filename,

            "pages":
                pages,

            "chunks":
                total_stored
        }


    except Exception as e:

        print(
            "\n==================================="
        )

        print(
            "❌ PDF PROCESSING ERROR"
        )

        print(
            "Type:",
            type(e).__name__
        )

        print(
            "Message:",
            str(e)
        )

        print(
            "==================================="
        )


        return {

            "status": "error",

            "message":
                str(e),

            "pages":
                0,

            "chunks":
                0
        }


# ==========================================
# PDF VIEWER
# ==========================================

@app.get("/pdf/{filename}")
def get_pdf(filename: str):

    safe_filename = os.path.basename(
        filename
    )

    file_path = os.path.join(
        UPLOAD_FOLDER,
        safe_filename
    )

    if not os.path.exists(
        file_path
    ):

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

    use_pdf = bool(
        use_pdf
    )


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

        image_data = (
            image_data.strip()
        )

        if not image_data:

            image_data = None

        if (
            image_data
            and
            len(image_data) > 10_000_000
        ):

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

        question = str(
            question
        )

    question = (
        question.strip()
    )

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
        # LOAD RAG
        # ==================================

        print(
            "\n🤖 Loading RAG pipeline..."
        )

        rag_service = await run_in_threadpool(
            get_rag
        )

        print(
            "🤖 Sending request to RAG pipeline..."
        )


        # ==================================
        # ASK RAG
        # ==================================

        answer = await run_in_threadpool(
            rag_service.ask,
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
            type(e).__name__,
            str(e)
        )

        return {

            "status": "error",

            "message":
                str(e)
        }