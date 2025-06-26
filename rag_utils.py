# rag_utils.py
import re
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


INDEX_FOLDER_MAP = {
    "sales_data": "sales_data_sample_index",
    "sales_summary": "sales_summary_sample_index",
    "purchase_data": "purchase_data_sample_index",
    "pending_do": "pending_do_index",
    "ar_data": "ar_data_index",
    "ap_data": "ap_data_index"
}

INDEX_FILE_MAP = {
    "sales_data_sample_index": "sales_data_sample.csv",
    "sales_summary_sample_index": "sales_summary_sample.csv",
    "purchase_data_sample_index": "purchase_data_sample.csv",
    "pending_do_index": "pending_do.csv",
    "ar_data_index": "ar_data.csv",
    "ap_data_index": "ap_data.csv"
}

def get_index_from_question(question):
    q = question.lower()
    if "product" in q or "customer" in q or "salesman" in q:
        return INDEX_FOLDER_MAP["sales_data"]
    elif "summary" in q:
        return INDEX_FOLDER_MAP["sales_summary"]
    elif "supplier" in q or "purchase" in q:
        return INDEX_FOLDER_MAP["purchase_data"]
    elif "pending do" in q or "delivery order" in q:
        return INDEX_FOLDER_MAP["pending_do"]
    elif "receivable" in q or "ar" in q:
        return INDEX_FOLDER_MAP["ar_data"]
    elif "payable" in q or "ap" in q:
        return INDEX_FOLDER_MAP["ap_data"]
    return None

def load_data_from_index(index_folder):
    embeddings = OllamaEmbeddings(model="deepseek-llm:7b")
    vectorstore = FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)
    docs = vectorstore.docstore._dict.values()
    
    records = []
    for doc in docs:
        record = {}
        for part in doc.page_content.split(". "):
            if ": " in part:
                key, value = part.split(": ", 1)
                record[key.strip().lower().replace(" ", "_")] = value.strip()
        records.append(record)
    
    return pd.DataFrame(records)


def run_rag_fallback(question):
    index_name = get_index_from_question(question)
    if not index_name:
        return "🤖 Unable to find relevant index for your query."

    try:
        embeddings = OllamaEmbeddings(model="deepseek-llm:7b")
        vectorstore = FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        llm = OllamaLLM(model="deepseek-llm:7b", temperature=0.3)

        prompt = ChatPromptTemplate.from_template("""
        You are a Sales & Purchase Analyst Bot. Given context and a user question,
        return concise and structured answers **in bullet points** by filtering for:
        - Division (e.g., div_*)
        - Date filters (last 3/6 months, Jan, Feb, etc.)
        - Quantity or Revenue intent
        - Receivables/Payables/Aging/Profit/DO/Cash Flow
        Avoid hallucinations. Always refer to structured context.

        🧾 Context:
        {context}

        ❓ Question:
        {question}

        📌 Answer (as bullet points):
        """)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        result = chain.invoke(question)
        file_name = INDEX_FILE_MAP.get(index_name, "Unknown file")
        return f"{result}\n\n📂 Source Index: {index_name}\n📄 Backed by File: {file_name}"

    except Exception as e:
        return f"❌ Failed to run RAG fallback: {e}"
