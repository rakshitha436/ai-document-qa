# ============================================================
# main.py
# THE MAIN ENTRY POINT — runs the full RAG pipeline
# ============================================================

import os
import sys
from pdf_loader import load_pdf
from text_chunker import split_into_chunks
from embeddings import create_vector_store, save_vector_store, load_vector_store
from qa_pipeline import build_qa_chain, ask_question

# ── FAISS index folder name (where the vector DB is saved on disk) ────────────
FAISS_INDEX_PATH = "faiss_index"


def process_pdf(pdf_path: str):
    """
    Full pipeline: PDF → chunks → embeddings → FAISS index
    Run this once per document. The index is saved so you don't re-process.
    """
    
    print("=" * 55)
    print("   📘 RAG Document QA System — Processing PDF")
    print("=" * 55)
    
    # STEP 1: Load the PDF
    pages = load_pdf(pdf_path)
    
    # STEP 2: Split into chunks
    chunks = split_into_chunks(pages, chunk_size=500, chunk_overlap=50)
    
    # STEP 3 & 4: Embed chunks and store in FAISS
    vector_store = create_vector_store(chunks)
    
    # Save to disk so we can reload without re-processing next time
    save_vector_store(vector_store, FAISS_INDEX_PATH)
    
    return vector_store


def interactive_qa(vector_store):
    """
    Start an interactive Q&A loop in the terminal.
    The user types questions, gets answers, until they type 'exit'.
    """
    
    print("\n" + "=" * 55)
    print("   💬 Interactive Q&A — Ask about your document")
    print("   Type 'exit' or 'quit' to stop")
    print("=" * 55)
    
    # Build the LangChain QA pipeline
    qa_chain = build_qa_chain(vector_store)
    
    # Keep looping until user exits
    while True:
        print()
        user_question = input("🧑 Your question: ").strip()
        
        # Exit condition
        if user_question.lower() in ["exit", "quit", "q"]:
            print("\n👋 Goodbye! Your FAISS index is saved for next time.")
            break
        
        # Skip empty input
        if not user_question:
            print("⚠️  Please enter a question.")
            continue
        
        # Run the RAG pipeline and print the answer
        ask_question(qa_chain, user_question)
        print("\n" + "-" * 55)


def main():
    """
    Main function — decides whether to process a new PDF
    or load an existing saved index.
    """
    
    # ── Check for OpenAI API key ──────────────────────────────────────────────
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable is not set.")
        print("   Run: export OPENAI_API_KEY='sk-your-key-here'")
        sys.exit(1)
    
    # ── Decide: process new PDF or load existing index ────────────────────────
    if len(sys.argv) > 1:
        # A PDF path was passed as a command-line argument
        # Usage: python main.py path/to/your/file.pdf
        pdf_path = sys.argv[1]
        vector_store = process_pdf(pdf_path)
        
    elif os.path.exists(FAISS_INDEX_PATH):
        # No PDF argument, but a saved index exists — load it
        print(f"🔍 Found existing FAISS index at './{FAISS_INDEX_PATH}/'")
        print("   Loading saved index (skipping PDF processing)...")
        vector_store = load_vector_store(FAISS_INDEX_PATH)
        
    else:
        # No PDF and no saved index — show usage instructions
        print("❌ No PDF file provided and no saved index found.")
        print("\nUsage:")
        print("   python main.py path/to/your/document.pdf")
        print("\nExample:")
        print("   python main.py resume.pdf")
        sys.exit(1)
    
    # ── Start the interactive Q&A loop ───────────────────────────────────────
    interactive_qa(vector_store)


if __name__ == "__main__":
    main()
