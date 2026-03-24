from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

def build_qa_chain(vector_store: FAISS):
    print("\n🤖 Building the QA pipeline...")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    prompt_template = """You are a helpful assistant that answers questions based ONLY 
on the provided document context. If the answer is not found in the context, 
say "I couldn't find that information in the document."

Context from the document:
{context}

User Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    print("✅ QA pipeline ready!")
    return qa_chain


def ask_question(qa_chain, question: str):
    print(f"\n❓ Question: {question}")
    print("⏳ Searching document and generating answer...")

    result = qa_chain.invoke({"query": question})

    answer = result["result"]
    source_docs = result["source_documents"]

    print(f"\n💬 Answer:\n{answer}")

    print(f"\n📚 Based on {len(source_docs)} document chunk(s):")
    for i, doc in enumerate(source_docs, 1):
        page_num = doc.metadata.get("page", "unknown")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"   Chunk {i} (page {page_num}): {preview}...")

    return answer
