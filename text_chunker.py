# ============================================================
# text_chunker.py
# STEP 2: Split large text into smaller, overlapping chunks
# ============================================================

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

def split_into_chunks(pages: List[Document], chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Split document pages into smaller text chunks.
    
    WHY DO WE CHUNK TEXT?
    LLMs have a limit on how much text they can process at once (called a "context window").
    Also, smaller chunks = more precise search results when we look for relevant info.
    
    Args:
        pages:         List of Document objects from pdf_loader.py
        chunk_size:    Max number of characters per chunk (default: 500)
        chunk_overlap: How many characters to repeat between chunks so we
                       don't lose meaning at the boundaries (default: 50)
    
    Returns:
        A list of smaller Document chunks
    """
    
    print(f"\n✂️  Splitting text into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    
    # RecursiveCharacterTextSplitter tries to split on paragraphs, then sentences,
    # then words — keeping splits as natural as possible
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # These separators are tried in order — it prefers paragraph breaks first
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split all pages into chunks
    chunks = splitter.split_documents(pages)
    
    print(f"✅ Created {len(chunks)} text chunk(s) from {len(pages)} page(s).")
    
    # Show a preview of the first chunk so you can see what it looks like
    if chunks:
        print(f"\n📝 Preview of first chunk:\n---\n{chunks[0].page_content[:200]}...\n---")
    
    return chunks
