from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pypdf import PdfReader
from docx import Document
from io import BytesIO

app = FastAPI(title="PDF/DOCX Text Extractor API")

def read_pdf_bytes(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()

def read_docx_bytes(file_bytes: bytes) -> str:
    doc = Document(BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text).strip()

@app.post(
    "/upload",
    status_code=status.HTTP_200_OK,
    summary="Upload PDF or DOCX and extract text"
)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File is required")

    filename = file.filename.lower()

    if not (filename.endswith(".pdf") or filename.endswith(".docx")):
        raise HTTPException(
            status_code=415,
            detail="Only PDF and DOCX files are allowed"
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if filename.endswith(".pdf"):
        extracted_text = read_pdf_bytes(file_bytes)
    else:  
        extracted_text = read_docx_bytes(file_bytes)

    if not extracted_text:
        raise HTTPException(
            status_code=422,
            detail="Text extraction failed (possibly scanned PDF)"
        )

    return {
        "filename": file.filename,
        "file_type": "pdf" if filename.endswith(".pdf") else "docx",
        "characters_extracted": len(extracted_text),
        "text": extracted_text
    }
