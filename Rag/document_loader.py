from pypdf import PdfReader


def load_pdf(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page_number, page in enumerate(reader.pages):
        page_text = page.extract_text()

        if page_text:
            text += page_text
        else:
            print(f"⚠️ Page {page_number + 1} has no extractable text")

    return text


if __name__ == "__main__":

    file_path = "data/sample.pdf"

    content = load_pdf(file_path)

    print("\n✅ PDF Loaded Successfully")
    print(f"Total characters extracted: {len(content)}")

    print("\n----- First 500 Characters -----\n")
    print(content[:500])