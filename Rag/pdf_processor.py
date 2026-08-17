"""
PDF Processor for DocChat AI

- Uses Docling for PDF parsing
- Optimized for Railway / low-memory environments
- Uses native PDF text extraction
- OCR disabled for normal text PDFs
- Table structure detection disabled
- Extracts structured text page by page
- Creates chunks while preserving page number and filename
- Output format remains compatible with Qdrant/RAG pipeline
"""

from typing import List, Dict, BinaryIO, Union
import io
import os

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFProcessor:

    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 150
    ):
        print("📖 Initializing lightweight Docling PDF processor...")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # =====================================================
        # LIGHTWEIGHT DOCLING CONFIGURATION
        # =====================================================

        pipeline_options = PdfPipelineOptions()

        # Normal text PDFs do not need OCR
        pipeline_options.do_ocr = False

        # Disable heavy table structure processing
        pipeline_options.do_table_structure = False

        # Use text already embedded inside the PDF
        pipeline_options.force_backend_text = True

        # Do not generate unnecessary images
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False
        pipeline_options.generate_table_images = False

        # Disable additional enrichment models
        pipeline_options.do_code_enrichment = False
        pipeline_options.do_formula_enrichment = False
        pipeline_options.do_picture_classification = False
        pipeline_options.do_picture_description = False
        pipeline_options.do_chart_extraction = False

        # =====================================================
        # DOCLING CONVERTER
        # =====================================================

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        # =====================================================
        # CHUNKING
        # =====================================================

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                " ",
                ""
            ]
        )

        print("✅ Lightweight Docling PDF processor ready.")

    # =========================================================
    # PREPARE PDF SOURCE
    # =========================================================

    def _prepare_source(
        self,
        pdf_file: Union[BinaryIO, bytes, str]
    ):
        """
        Prepare PDF input for Docling.

        Supports:
        - File path
        - bytes
        - file-like objects
        """

        if isinstance(pdf_file, str):
            return pdf_file

        if isinstance(pdf_file, bytes):
            return io.BytesIO(pdf_file)

        if hasattr(pdf_file, "read"):
            try:
                pdf_file.seek(0)
            except Exception:
                pass

            pdf_bytes = pdf_file.read()

            return io.BytesIO(pdf_bytes)

        raise TypeError(
            "Unsupported PDF input type."
        )

    # =========================================================
    # EXTRACT TEXT WITH PAGES
    # =========================================================

    def extract_text_with_pages(
        self,
        pdf_file: Union[BinaryIO, bytes, str]
    ) -> List[Dict]:
        """
        Extract text from PDF using lightweight Docling.

        Returns:

        [
            {
                "page": 1,
                "text": "..."
            },
            ...
        ]
        """

        source = self._prepare_source(pdf_file)

        print("📄 Processing PDF with lightweight Docling...")

        try:
            result = self.converter.convert(source)

        except Exception as e:
            raise ValueError(
                f"Docling could not process the PDF: {str(e)}"
            )

        document = result.document

        pages_data = []

        # =====================================================
        # PAGE-WISE EXTRACTION
        # =====================================================

        try:

            pages = document.pages

            for page_number, page in pages.items():

                try:
                    text = page.export_to_markdown()
                except Exception:
                    text = ""

                if text:
                    text = text.strip()

                if text:
                    pages_data.append(
                        {
                            "page": int(page_number),
                            "text": text
                        }
                    )

        except Exception:

            print(
                "⚠️ Page-wise extraction unavailable. "
                "Using document-level extraction."
            )

            try:
                full_text = document.export_to_markdown()
            except Exception:
                full_text = ""

            if full_text and full_text.strip():

                pages_data.append(
                    {
                        "page": 1,
                        "text": full_text.strip()
                    }
                )

        # =====================================================
        # VERIFY EXTRACTION
        # =====================================================

        if not pages_data:

            raise ValueError(
                "No text could be extracted from the PDF. "
                "The PDF may contain scanned/image-only pages."
            )

        print(
            f"✅ Docling extracted "
            f"{len(pages_data)} pages."
        )

        return pages_data

    # =========================================================
    # CREATE CHUNKS
    # =========================================================

    def create_chunks(
        self,
        pages_data: List[Dict],
        filename: str
    ) -> List[Dict]:
        """
        Split page-wise text into chunks.

        Keeps:
        - filename
        - page
        - chunk_index

        Output remains compatible with Qdrant.
        """

        chunks = []

        chunk_index = 0

        for page_info in pages_data:

            page_num = page_info["page"]
            page_text = page_info["text"]

            if not page_text:
                continue

            page_chunks = self.text_splitter.split_text(
                page_text
            )

            for chunk_text in page_chunks:

                if not chunk_text.strip():
                    continue

                chunks.append(
                    {
                        "content": chunk_text,

                        "metadata": {
                            "filename": filename,
                            "page": page_num,
                            "chunk_index": chunk_index
                        }
                    }
                )

                chunk_index += 1

        print(
            f"✂️ Created {len(chunks)} chunks."
        )

        return chunks

    # =========================================================
    # COMPLETE PDF PIPELINE
    # =========================================================

    def process_pdf(
        self,
        pdf_file: Union[BinaryIO, bytes, str],
        filename: str = "document.pdf"
    ) -> List[Dict]:
        """
        Complete PDF processing pipeline:

        1. Docling reads PDF
        2. Extract text page-by-page
        3. Split text into chunks
        4. Preserve metadata
        5. Return chunks for embeddings/Qdrant
        """

        print(
            f"\n📄 Processing: {filename}"
        )

        pages_data = self.extract_text_with_pages(
            pdf_file
        )

        chunks = self.create_chunks(
            pages_data,
            filename
        )

        if not chunks:

            raise ValueError(
                "PDF was read successfully, "
                "but no chunks were created."
            )

        print(
            f"✅ PDF processing completed: "
            f"{len(pages_data)} pages, "
            f"{len(chunks)} chunks."
        )

        return chunks


# =============================================================
# LOCAL TEST
# =============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage: python Rag/pdf_processor.py <pdf_path>"
        )

        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):

        print(
            f"❌ File not found: {pdf_path}"
        )

        sys.exit(1)

    processor = PDFProcessor()

    chunks = processor.process_pdf(
        pdf_path,
        filename=os.path.basename(pdf_path)
    )

    print("\n==============================")
    print("PDF TEST RESULT")
    print("==============================")

    print(
        f"Total chunks: {len(chunks)}"
    )

    for chunk in chunks[:3]:

        print("\n--- CHUNK ---")

        print(
            "Page:",
            chunk["metadata"]["page"]
        )

        print(
            "Content:",
            chunk["content"][:500]
        )