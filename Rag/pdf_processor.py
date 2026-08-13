"""
PDF Processor for DocChat AI
- Extracts text page by page using pdfplumber
- Recursive Character Text Splitting (chunk_size=900, overlap=150)
- Keeps page number + filename in metadata
"""

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, BinaryIO, Union
import io


class PDFProcessor:
    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )

    def extract_text_with_pages(self, pdf_file: Union[BinaryIO, bytes, str]) -> List[Dict]:
        """
        Extract text from every page of the PDF.
        Returns a list of dictionaries: {"page": int, "text": str}
        """
        pages_data = []

        # Handle different input types (Streamlit UploadedFile, bytes, or file path)
        if hasattr(pdf_file, "read"):
            # Streamlit UploadedFile or file-like object
            pdf_bytes = pdf_file.read()
            pdf_file.seek(0)  # Reset pointer for possible re-use
            source = io.BytesIO(pdf_bytes)
        elif isinstance(pdf_file, bytes):
            source = io.BytesIO(pdf_file)
        else:
            # Assume it is a file path
            source = pdf_file

        with pdfplumber.open(source) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = text.strip()
                if text:  # Skip completely empty pages
                    pages_data.append({
                        "page": page_num,
                        "text": text
                    })

        if not pages_data:
            raise ValueError("No text could be extracted from the PDF. It might be a scanned PDF (OCR required).")

        return pages_data

    def create_chunks(self, pages_data: List[Dict], filename: str) -> List[Dict]:
        """
        Split page-wise text into overlapping chunks.
        Each chunk keeps page number and filename in metadata.
        """
        chunks = []
        chunk_index = 0

        for page_info in pages_data:
            page_num = page_info["page"]
            page_text = page_info["text"]

            # Split this page's text into smaller chunks
            page_chunks = self.text_splitter.split_text(page_text)

            for chunk_text in page_chunks:
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "filename": filename,
                        "page": page_num,
                        "chunk_index": chunk_index
                    }
                })
                chunk_index += 1

        return chunks

    def process_pdf(self, pdf_file: Union[BinaryIO, bytes, str], filename: str = "document.pdf") -> List[Dict]:
        """
        Complete pipeline:
        1. Extract text page-by-page
        2. Create recursive chunks with metadata
        Returns a list of chunks ready for embedding.
        """
        pages_data = self.extract_text_with_pages(pdf_file)
        chunks = self.create_chunks(pages_data, filename)
        return chunks