# rag_core/rag/pdf_loader.py

from pypdf import PdfReader


def load_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text