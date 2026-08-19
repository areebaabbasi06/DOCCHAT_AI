"""
PDF Processor for DocChat AI

- Uses Docling for lightweight PDF parsing
- OCR disabled to reduce memory usage
- Heavy table/image processing disabled
- Extracts text page-by-page
- Preserves actual PDF page numbers
- Creates chunks with filename + page metadata
- Output remains compatible with Qdrant/RAG pipeline
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

        print(
            "📖 Initializing lightweight Docling PDF processor..."
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # =====================================================
        # LIGHTWEIGHT DOCLING PDF CONFIGURATION
        # =====================================================

        pipeline_options = PdfPipelineOptions()

        # Disable Torch compilation.
        pipeline_options.layout_options.engine_options.compile_model = False

        # OCR disabled for normal text PDFs.
        pipeline_options.do_ocr = False

        # Disable heavy table processing.
        pipeline_options.do_table_structure = False

        # Use embedded/native PDF text.
        pipeline_options.force_backend_text = True

        # Do not generate images.
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False
        pipeline_options.generate_table_images = False

        # Disable optional enrichment features.
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

        print(
            "✅ Lightweight Docling PDF processor ready."
        )

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

            if not pdf_bytes:

                raise ValueError(
                    "The uploaded PDF file is empty."
                )

            return io.BytesIO(pdf_bytes)

        raise TypeError(
            "Unsupported PDF input type."
        )

    # =========================================================
    # EXTRACT TEXT WITH ACTUAL PAGE NUMBERS
    # =========================================================

    def extract_text_with_pages(
        self,
        pdf_file: Union[BinaryIO, bytes, str]
    ) -> List[Dict]:

        """
        Extract text from PDF using Docling while
        preserving actual PDF page numbers.

        Each PDF page becomes a separate text section.
        """

        source = self._prepare_source(
            pdf_file
        )

        print(
            "📄 Processing PDF with lightweight Docling..."
        )

        try:

            result = self.converter.convert(
                source
            )

        except Exception as e:

            print(
                f"❌ Docling PDF processing error: {e}"
            )

            raise ValueError(
                f"Docling could not process the PDF: {str(e)}"
            )

        document = result.document

        pages_data = []

        # =====================================================
        # PAGE-WISE TEXT EXTRACTION
        # =====================================================

        try:

            # -------------------------------------------------
            # Docling stores document items with page numbers.
            # We collect text item-by-item and group them
            # according to their actual PDF page.
            # -------------------------------------------------

            page_texts = {}

            for item in document.iterate_items():

                try:

                    text = getattr(
                        item,
                        "text",
                        None
                    )

                    if not text:

                        continue

                    text = str(
                        text
                    ).strip()

                    if not text:

                        continue

                    # -----------------------------------------
                    # Get page number from item's provenance
                    # -----------------------------------------

                    page_number = None

                    provenance = getattr(
                        item,
                        "prov",
                        None
                    )

                    if provenance:

                        try:

                            for prov_item in provenance:

                                page_no = getattr(
                                    prov_item,
                                    "page_no",
                                    None
                                )

                                if page_no is not None:

                                    page_number = int(
                                        page_no
                                    )

                                    break

                        except Exception:
                            page_number = None

                    # -----------------------------------------
                    # Fallback
                    # -----------------------------------------

                    if page_number is None:

                        page_number = 1

                    if page_number not in page_texts:

                        page_texts[
                            page_number
                        ] = []

                    page_texts[
                        page_number
                    ].append(
                        text
                    )

                except Exception as item_error:

                    print(
                        "⚠️ Could not read one "
                        f"document item: {item_error}"
                    )

                    continue


            # =================================================
            # BUILD PAGE DATA
            # =================================================

            for page_number in sorted(
                page_texts.keys()
            ):

                page_text = "\n".join(
                    page_texts[
                        page_number
                    ]
                ).strip()

                if not page_text:

                    continue

                pages_data.append(
                    {
                        "page": page_number,
                        "text": page_text
                    }
                )


        except Exception as e:

            print(
                "⚠️ Page-wise extraction failed."
            )

            print(
                f"Reason: {e}"
            )


        # =====================================================
        # FALLBACK TO DOCUMENT MARKDOWN
        # =====================================================

        if not pages_data:

            print(
                "⚠️ Page metadata was not available."
            )

            print(
                "➡️ Using document-level text fallback..."
            )

            try:

                full_text = (
                    document.export_to_markdown()
                )

                if full_text:

                    full_text = (
                        full_text.strip()
                    )

                if full_text:

                    pages_data.append(
                        {
                            "page": 1,
                            "text": full_text
                        }
                    )

            except Exception as e:

                print(
                    "❌ Document text extraction failed:"
                )

                print(
                    e
                )


        # =====================================================
        # VERIFY EXTRACTION
        # =====================================================

        if not pages_data:

            raise ValueError(
                "No text could be extracted from the PDF. "
                "The PDF may be scanned/image-only or "
                "may not contain extractable text."
            )


        print(
            "✅ Document text extracted successfully."
        )

        print(
            f"📄 Actual pages with text: "
            f"{len(pages_data)}"
        )


        # =====================================================
        # SHOW PAGE RANGE
        # =====================================================

        page_numbers = [
            page["page"]
            for page in pages_data
        ]

        if page_numbers:

            print(
                f"📑 Page numbers detected: "
                f"{min(page_numbers)}-"
                f"{max(page_numbers)}"
            )


        print(
            f"✅ Docling extracted "
            f"{len(pages_data)} page sections."
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
        Split each PDF page into chunks.

        Keeps:
        - filename
        - actual page number
        - chunk_index
        """

        chunks = []

        chunk_index = 0

        for page_info in pages_data:

            page_num = page_info[
                "page"
            ]

            page_text = page_info[
                "text"
            ]

            if not page_text:

                continue

            page_chunks = (
                self.text_splitter.split_text(
                    page_text
                )
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
        3. Preserve actual page numbers
        4. Split text into chunks
        5. Preserve metadata
        6. Return chunks for embeddings/Qdrant
        """

        print(
            f"\n📄 Processing: {filename}"
        )

        pages_data = (
            self.extract_text_with_pages(
                pdf_file
            )
        )

        chunks = (
            self.create_chunks(
                pages_data,
                filename
            )
        )

        if not chunks:

            raise ValueError(
                "PDF was read successfully, "
                "but no chunks were created."
            )


        print(
            f"✅ PDF processing completed: "
            f"{len(pages_data)} page sections, "
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

    if not os.path.exists(
        pdf_path
    ):

        print(
            f"❌ File not found: {pdf_path}"
        )

        sys.exit(1)


    processor = PDFProcessor()


    try:

        chunks = (
            processor.process_pdf(
                pdf_path,
                filename=os.path.basename(
                    pdf_path
                )
            )
        )


        print(
            "\n=============================="
        )

        print(
            "PDF TEST RESULT"
        )

        print(
            "=============================="
        )


        print(
            f"Total chunks: {len(chunks)}"
        )


        # Show first 3 chunks
        for chunk in chunks[:3]:

            print(
                "\n--- CHUNK ---"
            )

            print(
                "Page:",
                chunk[
                    "metadata"
                ][
                    "page"
                ]
            )

            print(
                "Content:",
                chunk[
                    "content"
                ][:500]
            )


        # Show unique pages
        unique_pages = sorted(
            set(
                chunk[
                    "metadata"
                ][
                    "page"
                ]
                for chunk in chunks
            )
        )


        print(
            "\n=============================="
        )

        print(
            f"Pages represented in chunks: "
            f"{len(unique_pages)}"
        )

        print(
            f"Page numbers: {unique_pages}"
        )

        print(
            "=============================="
        )


    except Exception as e:

        print(
            "\n=============================="
        )

        print(
            "❌ PDF TEST FAILED"
        )

        print(
            "=============================="
        )

        print(
            e
        )

        sys.exit(1)