from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
import os


def build_qa_chain(vector_store: FAISS):
    print("\n🤖 Building the QA pipeline...")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    prompt = PromptTemplate.from_template("""You are a helpful assistant that answers questions based ONLY on the provided document context. If the answer is not found in the context, say "I couldn't find that information in the document."

Context from the document:
{context}

User Question: {question}

Answer:""")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ QA pipeline ready!")
    return qa_chain


def ask_question(qa_chain, question: str):
    print(f"\n❓ Question: {question}")
    print("⏳ Searching document and generating answer...")

    answer = qa_chain.invoke(question)

    print(f"\n💬 Answer:\n{answer}")
    return answer
