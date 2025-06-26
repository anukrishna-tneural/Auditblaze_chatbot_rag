# utils/intent_classifier.py
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

# ────────────────────────────────────────────────────────────
# 🔧  Model settings
# ────────────────────────────────────────────────────────────
llm = OllamaLLM(model="deepseek-llm:7b", temperature=0)

# ────────────────────────────────────────────────────────────
# 🏷️  Valid labels – must exactly match keys in rag_app.handler_map
# ────────────────────────────────────────────────────────────
LABELS = """
- top_products          (e.g. “best-selling items this month”)
- top_customers         (e.g. “biggest clients by revenue”)
- top_suppliers         (e.g. “highest-value vendors”)
- sales_by_salesman     (e.g. “who’s my top salesperson?”)
- receivables           (AR aging, overdue customers…)
- payables              (AP aging, overdue suppliers…)
- pending_do            (open / pending Delivery Orders)
- profit                (profitability / margin questions)
- cash_flow             (cash-flow forecast, inflow/outflow)
- greeting              (hi, hello…)
- thanks                (thanks, appreciate it…)
- help                  (what can you do?)
- unknown               (everything else)
""".strip()

PROMPT = PromptTemplate.from_template(
    f"""
You are an **intent classifier** for a sales-analytics chatbot.
Choose the single best label from the list below and output **ONLY** that label
(without extra text, punctuation, or code-blocks).

Valid labels:
{LABELS}

User question: {{question}}
Label:
"""
)

_chain = PROMPT | llm | StrOutputParser()

def classify_intent(question: str) -> str:
    """
    Return the lowercase label string, stripped of whitespace.
    """
    return _chain.invoke({"question": question}).strip().lower()
