import re
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from query_handlers import handle_query_intent
from rag_utils import get_index_from_question, INDEX_FILE_MAP


class SalesRAG:
    def __init__(self):
        pass  # Optional for future use

    def run_rag_fallback(self, question):
        index_name = get_index_from_question(question)
        if not index_name:
            return "🤖 Unable to find relevant index for your query."

        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectorstore = FAISS.load_local(
                index_name,
                embeddings,
                allow_dangerous_deserialization=True
            )
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

    def handle_quick(self, message):
        greetings = ["hi", "hello", "hey"]
        thanks = ["thank you", "thanks"]
        help_qs = ["help", "what can you do", "capabilities"]

        msg = message.lower().strip()
        print(f"🟡 Quick handler checking message: {msg}")
        if any(msg.startswith(g) for g in greetings):
            return "👋 Hello! Ask me about sales, suppliers, receivables, cash flow and more!"
        if any(t in msg for t in thanks):
            return "😊 You're welcome! Let me know if you need anything else."
        if any(h in msg for h in help_qs):
            return (
                "📌 I can help with queries like:\n"
                "- Top products/customers/salesmen (by revenue or quantity)\n"
                "- Receivables, payables, pending DOs\n"
                "- Supplier performance and profitability\n"
                "- Cash flow forecast for next 10 days"
            )
        return None

    def query_handler(self, input_text):
        quick = self.handle_quick(input_text)
        if quick:
            return quick

        try:
            structured_response = handle_query_intent(input_text)
            if structured_response:  # <-- ADD THIS CHECK
                return structured_response
            else:
                return self.run_rag_fallback(input_text)
        except Exception:
            return self.run_rag_fallback(input_text)


    # ✅ Add shim for Flask and CLI compatibility
    def query(self, message):
        return self.query_handler(message)


# CLI entry point (optional for testing)
if __name__ == "__main__":
    bot = SalesRAG()
    print("🤖 RAG Chatbot ready. Type your query or 'exit'.")
    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            break
        response = bot.query(q)
        print(f"Bot: {response}\n")
