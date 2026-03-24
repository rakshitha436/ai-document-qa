from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def split_into_chunks(pages: List[Document], chunk_size: int = 500, chunk_overlap: int = 50):
    print(f"\n✂️  Splitting text into chunks (size={chunk_size}, overlap={chunk_overlap})...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(pages)
    print(f"✅ Created {len(chunks)} text chunk(s) from {len(pages)} page(s).")

    if chunks:
        print(f"\n📝 Preview of first chunk:\n---\n{chunks[0].page_content[:200]}...\n---")

    return chunks
