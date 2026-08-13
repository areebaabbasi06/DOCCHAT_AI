import os


MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


def validate_file(file_path):

    # Check file exists
    if not os.path.exists(file_path):
        return False, "File does not exist"


    # Check extension
    if not file_path.lower().endswith(".pdf"):
        return False, "Only PDF files are allowed"


    # Check size
    file_size = os.path.getsize(file_path)

    if file_size > MAX_FILE_SIZE:
        return False, "File size exceeds 5MB limit"


    return True, "File is valid"



if __name__ == "__main__":

    path = "data/sample.pdf"

    status, message = validate_file(path)

    print(status)
    print(message)