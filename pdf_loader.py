# ============================================================
# pdf_loader.py
# STEP 1: Load and extract text from a PDF file
# ============================================================

from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdf(file_path: str):
    """
    Load a PDF file and extract its text content.
    
    Args:
        file_path: The path to the PDF file on your computer
        
    Returns:
        A list of Document objects, one per page
    """
    
    # Check if the file actually exists before trying to load it
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
    
    print(f"📄 Loading PDF: {file_path}")
    
    # PyPDFLoader reads the PDF and splits it into pages automatically
    # Each page becomes a "Document" object with .page_content and .metadata
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    print(f"✅ Successfully loaded {len(pages)} page(s) from the PDF.")
    
    return pages
