def split_text(text, chunk_size=500, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

    return chunks


if __name__ == "__main__":

    from document_loader import load_pdf

    file_path = "data/sample.pdf"

    text = load_pdf(file_path)

    chunks = split_text(text)

    print("✅ Text Split Successfully")
    print(f"Total chunks created: {len(chunks)}")

    print("\n----- First Chunk -----\n")
    print(chunks[0])