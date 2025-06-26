import re
import pandas as pd
from datetime import datetime, timedelta
from rag_utils import get_index_from_question, load_data_from_index, run_rag_fallback

# --------------------------------------------------
# 📅 Constants
# --------------------------------------------------
MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

# --------------------------------------------------
# ✨ Helper utilities
# --------------------------------------------------
WORD_RE = re.compile(r"\b\w+\b")  # reuse compiled regex

def tokens(q: str) -> set[str]:
    """Return lowercase word tokens from question."""
    return set(WORD_RE.findall(q.lower()))

def _metric_flags(q: str):
    q = q.lower()
    rev = any(k in q for k in ["revenue", "net amount", "netamount", "sales amount", "amount", "value"])
    qty = any(k in q for k in ["quantity", "qty", "volume"])
    return rev, qty

def _intro(entity: str, top_n: int, q: str, asc: bool) -> str:
    rev, qty = _metric_flags(q)
    order_word = "lowest" if asc else "highest"
    if rev and not qty:
        metric = "revenue"
    elif qty and not rev:
        metric = "quantity sold"
    else:
        metric = "both revenue and quantity"
    return f"🧠 You asked for the {top_n} {order_word}-performing {entity} by {metric}.\n\n"

# --------------------------------------------------
# 🔎 Filtering
# --------------------------------------------------
def apply_division_filter(df, divisions):
    if not divisions:
        return df
    divisions = [d.strip().lower() for d in divisions if pd.notna(d)]
    if 'division' not in df.columns:
        return df
    filtered = df[df['division'].str.lower().isin(divisions)]
    if filtered.empty:
        available = df['division'].dropna().unique().tolist()
        return f"⚠️ No data for divisions {divisions}. Available: {available}"
    return filtered

def apply_date_filter(df, question):
    # 👉 Bail out early if docdate column is missing
    if 'docdate' not in df.columns:
        return df

    df['docdate'] = pd.to_datetime(df['docdate'], errors='coerce')
    now = datetime.today()
    q = question.lower()

    if "last quarter" in q or "quarter" in q:
        return df[df['docdate'] >= now - timedelta(days=120)] 
    elif "this month" in q:                     
        start = now.replace(day=1)
        return df[df['docdate'] >= start]
    elif "last 6 months" in q:
        return df[df['docdate'] >= now - timedelta(days=180)]
    elif "last 3 months" in q:
        return df[df['docdate'] >= now - timedelta(days=90)]
    elif "last month" in q:
        start = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
        return df[df['docdate'] >= start]

    m = re.search(r"last (\d+) months?", q)
    if m:
        start = now - pd.DateOffset(months=int(m.group(1)))
        return df[df['docdate'] >= start]

    for mon in MONTH_MAP:
        if mon in q:
            return df[
                (df['docdate'].dt.month == MONTH_MAP[mon])
                & (df['docdate'].dt.year == 2024)  # 🔁 default to last year
            ]

    # 👇 Default to entire year 2024
    return df[df['docdate'].dt.year == 2024]


# --------------------------------------------------
# 🧠 Intent helpers
# --------------------------------------------------
def tokens(q: str) -> set[str]:
    """Return lowercase word tokens from question."""
    return set(re.findall(r"\b\w+\b", q.lower()))

def is_greeting(q: str) -> bool:
    return {"hi", "hello", "hey"} & tokens(q)

def extract_divisions_from_question(question):
    return re.findall(r"div_\d+", question.lower())

def guess_index_from_keywords(question: str):
    t = tokens(question)
    q_lower = question.lower()

    if any(p in q_lower for p in [
        "pending do", "do to invoice", "delivery order",
        "open do", "pending delivery"
    ]):
        return "pending_do_index"

    # ✅ Payables with aging support
    if {"payable", "payables", "ap", "outstanding", "aging"} & t:
        return "ap_data_index"

    # ✅ Receivables
    if {"receivable", "receivables", "overdue", "aging", "ageing"} & t or "ar" in t:
        return "ar_data_index"

    if {"supplier", "suppliers", "vendor", "vendors"} & t:
        return "purchase_data_sample_index"

    if {"product", "item", "stock", "bestseller", "sales",
        "customer", "client", "salesman", "margin"} & t:
        return "sales_data_sample_index"

    return None


def determine_sort_order(question):
    return any(k in question.lower() for k in ["worst", "least", "bottom", "lowest", "low"])

# --------------------------------------------------
# 🔧 Shared Handler Logic
# --------------------------------------------------
def format_dual_ranking(df, group_field, result_label, question, top_n=5, asc=False):
    rev_flag, qty_flag = _metric_flags(question)
    result = _intro(result_label, top_n, question, asc)
    grouped = df.groupby(group_field)[["netamount", "quantity"]].sum().reset_index()

    if rev_flag and qty_flag:
        top_rev = grouped.sort_values("netamount", ascending=asc).head(top_n)
        top_qty = grouped.sort_values("quantity", ascending=asc).head(top_n)
        result += "💰 Top by Revenue:\n"
        for _, r in top_rev.iterrows():
            result += f"• {r[group_field]} → AED {r['netamount']:.2f}, Qty {int(r['quantity'])}\n"
        result += "\n📦 Top by Quantity:\n"
        for _, r in top_qty.iterrows():
            result += f"• {r[group_field]} → Qty {int(r['quantity'])}, AED {r['netamount']:.2f}\n"
        result += "\n📌 Note: Rankings differ for revenue vs quantity."
        return result
    elif qty_flag:
        top_qty = grouped.sort_values("quantity", ascending=asc).head(top_n)
        result += "📦 By Quantity:\n"
        for _, r in top_qty.iterrows():
            result += f"• {r[group_field]} → Qty {int(r['quantity'])}, AED {r['netamount']:.2f}\n"
        return result
    else:
        top_rev = grouped.sort_values("netamount", ascending=asc).head(top_n)
        result += "💰 By Revenue:\n"
        for _, r in top_rev.iterrows():
            result += f"• {r[group_field]} → AED {r['netamount']:.2f}, Qty {int(r['quantity'])}\n"
        return result

# --------------------------------------------------
# 🎯 Main Handlers
# --------------------------------------------------
def handle_sales_products(df, question, divisions=None, top_n=5):
    df = apply_division_filter(df, divisions)
    if isinstance(df, str): return df
    df = apply_date_filter(df, question)
    if df.empty: return "No product data available for the given filters."

    # 🩹 Default to both revenue and quantity if not mentioned
    if not any(k in question.lower() for k in ["revenue", "amount", "value", "quantity", "qty", "volume"]):
        question += " revenue and quantity"

    return format_dual_ranking(df, "stockdescription", "products", question, top_n, determine_sort_order(question))

def handle_customers(df, question, divisions=None, top_n=5):
    df = apply_division_filter(df, divisions)
    if isinstance(df, str): return df
    df = apply_date_filter(df, question)
    if df.empty: return "No customer data available."

    # 🩹 Default to both revenue and quantity if not mentioned
    if not any(k in question.lower() for k in ["revenue", "amount", "value", "quantity", "qty", "volume"]):
        question += " revenue and quantity"

    return format_dual_ranking(df, "partyname", "customers", question, top_n, determine_sort_order(question))


def handle_salesmen(df, question, divisions=None, top_n=5):
    df = apply_division_filter(df, divisions)
    if isinstance(df, str): return df
    df = apply_date_filter(df, question)
    if df.empty: return "No salesman data available."

    # 🔧 Patch: If neither "revenue" nor "quantity" is mentioned, assume both
    if not any(k in question.lower() for k in ["revenue", "amount", "value", "quantity", "qty", "volume"]):
        question += " revenue and quantity"

    return format_dual_ranking(df, "salesman", "salesmen", question, top_n, determine_sort_order(question))


# --------------------------------------------------
# 🧾  Supplier handler (supports brands filtering & comparison)
# --------------------------------------------------
def extract_brands_from_question(question: str):
    return re.findall(r"brand_\d+", question.lower())


def handle_top_suppliers(df, question: str, top_n: int = 5):
    asc         = determine_sort_order(question)
    order_word  = "lowest" if asc else "highest"
    brands_q    = extract_brands_from_question(question)
    divisions_q = extract_divisions_from_question(question)
    rev_flag, qty_flag = _metric_flags(question)

    # Normalize columns
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    net_amt_col = next((c for c in ['net_amount', 'netamount', 'net amount'] if c in df.columns), None)
    if not net_amt_col:
        return f"❌ Cannot find a Net Amount column. Found columns: {df.columns.tolist()}"
    df['net_amount'] = pd.to_numeric(df[net_amt_col], errors='coerce')

    if 'party_name' not in df.columns:
        return "❌ Required field 'party_name' not found."

    df['quantity'] = pd.to_numeric(df.get('quantity', 0), errors='coerce')

    if 'docdate' in df.columns:
        df['docdate'] = pd.to_datetime(df['docdate'], errors='coerce')
        df = apply_date_filter(df, question)

    # Optional filters: brand, division
    brand_col = next((c for c in ['product_brand', 'brand'] if c in df.columns), None)
    div_col   = next((c for c in ['product_division', 'division'] if c in df.columns), None)
    if brand_col: df[brand_col] = df[brand_col].astype(str).str.lower()
    if div_col: df[div_col] = df[div_col].astype(str).str.lower()

    # Clean data
    df = df.dropna(subset=['net_amount', 'quantity'])

    # Composite Filter
    filters = []
    if brands_q and brand_col:
        filters.append(df[brand_col].isin([b.lower() for b in brands_q]))
    if divisions_q and div_col:
        filters.append(df[div_col].isin([d.lower() for d in divisions_q]))

    if filters:
        combined = filters[0]
        for f in filters[1:]:
            combined &= f
        df = df[combined]
        if df.empty:
            return "⚠️ No data found for the specified filters."

    # Group and Rank
    grp = df.groupby('party_name').sum(numeric_only=True).reset_index()

    if grp.empty:
        return f"⚠️ No suppliers found for the given filters: " + \
            f"{'brands: ' + ', '.join(brands_q) if brands_q else ''}" 


    out = f"🏆 {top_n} {order_word.title()} Suppliers"
    if brands_q: out += f" for brand(s): {', '.join(brands_q)}"
    if divisions_q: out += f" in division(s): {', '.join(divisions_q)}"
    out += "\n\n"

    if rev_flag and not qty_flag:
        ranked = grp.sort_values('net_amount', ascending=asc).head(top_n)
        out += "💰 By Purchase Value:\n"
        for _, r in ranked.iterrows():
            out += f"• {r['party_name']} → AED {r['net_amount']:,.2f}, Qty {int(r['quantity'])}\n"

    elif qty_flag and not rev_flag:
        ranked = grp.sort_values('quantity', ascending=asc).head(top_n)
        out += "📦 By Quantity Supplied:\n"
        for _, r in ranked.iterrows():
            out += f"• {r['party_name']} → Qty {int(r['quantity'])}, AED {r['net_amount']:,.2f}\n"

    else:
        by_val = grp.sort_values('net_amount', ascending=asc).head(top_n)
        by_qty = grp.sort_values('quantity', ascending=asc).head(top_n)

        out += "💰 Top by Purchase Value:\n"
        for _, r in by_val.iterrows():
            out += f"• {r['party_name']} → AED {r['net_amount']:,.2f}, Qty {int(r['quantity'])}\n"

        out += "\n📦 Top by Quantity Supplied:\n"
        for _, r in by_qty.iterrows():
            out += f"• {r['party_name']} → Qty {int(r['quantity'])}, AED {r['net_amount']:,.2f}\n"

    return out

# --------------------------------------------------
# 💰  Profitable-products handler
# --------------------------------------------------
def handle_profitable_products(sales_df: pd.DataFrame,
                               purchase_df: pd.DataFrame,
                               question: str,
                               top_n: int = 5,
                               asc: bool = None):
    """
    Computes top / bottom N profitable products by margin % or profit.
    Now includes:
    - Unit cost and sale price
    - 🚨 loss-making product flag
    """

    # ---- Normalise column names
    for df in (sales_df, purchase_df):
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required_sales = ['stockcode', 'stockdescription', 'quantity', 'netamount']
    missing = [c for c in required_sales if c not in sales_df.columns]
    if missing:
        return f"❌ Missing sales columns: {', '.join(missing)}"
    if 'stock_code' not in purchase_df.columns or 'rate' not in purchase_df.columns:
        return "❌ Purchase data must have 'stock_code' and 'rate'."

    # ---- Clean and filter
    sales_df['quantity']  = pd.to_numeric(sales_df['quantity'],  errors='coerce')
    sales_df['netamount'] = pd.to_numeric(sales_df['netamount'], errors='coerce')
    purchase_df['rate']   = pd.to_numeric(purchase_df['rate'],   errors='coerce')

    sales_df.dropna(subset=['stockcode', 'quantity', 'netamount'], inplace=True)
    purchase_df.dropna(subset=['stock_code', 'rate'], inplace=True)

    sales_df    = apply_date_filter(sales_df,    question)
    purchase_df = apply_date_filter(purchase_df, question)

    if sales_df.empty or purchase_df.empty:
        return "⚠️ No sales or purchase data left after applying the date filter."

    # ---- Aggregate sales
    sales_agg = (
        sales_df
        .groupby(['stockcode', 'stockdescription'], as_index=False)
        .agg(total_sales=('netamount', 'sum'),
             total_qty  =('quantity',  'sum'))
    )

    # ---- Compute sale price per unit
    sales_agg['unit_sale_price'] = sales_agg['total_sales'] / sales_agg['total_qty']

    # ---- Average cost per product
    cost_agg = (
        purchase_df
        .groupby('stock_code', as_index=False)
        .agg(avg_cost=('rate', 'mean'))
    )

    # ---- Merge and calculate profitability
    merged = (
        sales_agg
        .merge(cost_agg, left_on='stockcode', right_on='stock_code', how='left')
        .drop(columns=['stock_code'])
    )

    merged['total_cost'] = merged['avg_cost'] * merged['total_qty']
    merged['profit']     = merged['total_sales'] - merged['total_cost']
    merged['margin_pct'] = (merged['profit'] / merged['total_sales']) * 100
    merged.dropna(subset=['profit', 'margin_pct'], inplace=True)

    if merged.empty:
        return "⚠️ Could not calculate profit margins due to missing cost data."

    # ---- Sort logic
    asc              = determine_sort_order(question)
    rev_flag, _      = _metric_flags(question)
    sort_field       = 'profit' if rev_flag else 'margin_pct'
    ranked           = merged.sort_values(sort_field, ascending=asc).head(top_n)
    order_word       = "lowest" if asc else "highest"

    # ---- Format output
    header = f"🏆 Top {top_n} {order_word}-margin products\n\n"
    body = ""
    for _, r in ranked.iterrows():
        alert = " 🚨" if r['margin_pct'] < 0 else ""
        body += (
            f"• {r['stockdescription']}{alert} → "
            f"Margin {r['margin_pct']:.1f}%, "
            f"Profit AED {r['profit']:,.2f}, "
            f"Sales AED {r['total_sales']:,.2f}, "
            f"Unit Cost AED {r['avg_cost']:.2f}, "
            f"Unit Price AED {r['unit_sale_price']:.2f}\n"
        )
    return header + body

def extract_entities_from_question(question: str, field: str):
    """
    Extract entities like salesman_123, cust_456, etc.
    """
    return re.findall(rf"{field.lower()}_\w+", question.lower())



# -------------------------------------------------
# 2️⃣  Pending-DO handler  –  entity‐aware version
# -------------------------------------------------
def handle_pending_do(df: pd.DataFrame, question: str, top_n: int = 10):

    # --- normalise ---
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if 'date' in df.columns and 'docdate' not in df.columns:
        df = df.rename(columns={'date': 'docdate'})
    if 'amount' not in df.columns:
        return "❌ Column 'amount' missing in pending_do data."
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)

    # --- entity extraction ---
    q_lower      = question.lower()
    salesmen_ids = extract_entities_from_question(q_lower, "salesman")  # e.g. ['salesman_083…']
    customers_ids= extract_entities_from_question(q_lower, "customer")

    # --- decide GROUP field ---
    # --- decide GROUP field ---
    explicit_customer = "customer" in q_lower
    explicit_salesman = "salesman" in q_lower

    if explicit_customer:
        group_field = "customer"
    elif explicit_salesman:
        group_field = "salesman"
    elif salesmen_ids:
        # 🔁 Default: If salesman mentioned but not grouping stated → group by customer
        group_field = "customer"
    else:
        group_field = "salesman"


    # --- division / date filters ---
    divisions = extract_divisions_from_question(question)
    df        = apply_division_filter(df, divisions)
    df        = apply_date_filter(df, question)

    # --- entity filters ---
    #   • If we’re grouping by customer, filter by any salesman IDs supplied
    #   • If we’re grouping by salesman, filter by any customer IDs supplied
    if salesmen_ids:
        ids = [s.replace("salesman_", "").replace("man_", "").lower() for s in salesmen_ids]
        df  = df[df['salesman'].astype(str).str.lower().isin(ids)]
    if customers_ids:
        ids = [c.replace("customer_", "").lower() for c in customers_ids]
        df  = df[df['customer'].astype(str).str.lower().isin(ids)]

    if df.empty:
        return "⚠️ No pending DOs match the requested filters."
    
    

    # --- aggregate ---
    grouped = (df.groupby(group_field, as_index=False)
                 .agg(total_pending=('amount', 'sum'))
                 .sort_values('total_pending', ascending=False)
                 .head(top_n))

    # --- format ---
    title = f"📦 Pending DOs – grouped by {group_field.title()}"
    if divisions:   title += f" | Divisions: {', '.join(divisions)}"
    if salesmen_ids: title+= f" | Salesman filter: {', '.join(salesmen_ids)}"
    if customers_ids:title+= f" | Customer filter: {', '.join(customers_ids)}"
    title += "\n\n"
    for _, r in grouped.iterrows():
        title += f"• {r[group_field]} → AED {r['total_pending']:,.2f}\n"
    return title



def handle_receivables_with_aging(df: pd.DataFrame, question: str, top_n: int = 10):
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Clean balance + aging
    df['balance'] = pd.to_numeric(df['balance'], errors='coerce')
    df.dropna(subset=['party_name', 'balance'], inplace=True)

    # Division filtering
    divisions = extract_divisions_from_question(question)
    df = apply_division_filter(df, divisions)
    if isinstance(df, str): return df
    if df.empty: return "⚠️ No receivables data for the given filters."

    # Aging buckets
    aging_cols = [
        'days_0_30', 'days_31_60', 'days_61_90', 'days_91_120',
        'days_121_150', 'days_151_180', 'days_181_210', 'days_above_210'
    ]
    for col in aging_cols:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)

    grouped = (
        df.groupby("party_name", as_index=False)[['balance'] + aging_cols]
          .sum()
          .sort_values("balance", ascending=False)
          .head(top_n)
    )

    # Output formatting
    out = f"🏦 Top {top_n} Customers with Receivables and Aging"
    if divisions: out += f" | Divisions: {', '.join(divisions)}"
    out += "\n\n"

    for _, r in grouped.iterrows():
        aging_str = " | ".join([
            f"0–30: {r['days_0_30']:,.0f}",
            f"31–60: {r['days_31_60']:,.0f}",
            f"61–90: {r['days_61_90']:,.0f}",
            f"91–120: {r['days_91_120']:,.0f}",
            f"121–150: {r['days_121_150']:,.0f}",
            f"151–180: {r['days_151_180']:,.0f}",
            f"181–210: {r['days_181_210']:,.0f}",
            f">210: {r['days_above_210']:,.0f}"
        ])
        out += f"• {r['party_name']} → AED {r['balance']:,.2f}\n  Aging: {aging_str}\n\n"

    return out


def handle_payables_with_aging(df: pd.DataFrame, question: str, top_n: int = 10):
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df['balance'] = pd.to_numeric(df['balance'], errors='coerce')
    df.dropna(subset=['party_name', 'balance'], inplace=True)

    divisions = extract_divisions_from_question(question)
    df = apply_division_filter(df, divisions)
    if isinstance(df, str): return df
    if df.empty: return "⚠️ No payables data for the given filters."

    aging_cols = [
        'days_0_30', 'days_31_60', 'days_61_90', 'days_91_120',
        'days_121_150', 'days_151_180', 'days_181_210', 'above_210'
    ]
    for col in aging_cols:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)

    grouped = (
        df.groupby("party_name", as_index=False)[['balance'] + aging_cols]
          .sum()
          .sort_values("balance", ascending=True)  # 👈 usually payable is negative
          .head(top_n)
    )

    out = f"🏦 Top {top_n} Suppliers with Payables and Aging"
    if divisions: out += f" | Divisions: {', '.join(divisions)}"
    out += "\n\n"

    for _, r in grouped.iterrows():
        aging_str = " | ".join([
            f"0–30: {r['days_0_30']:,.0f}",
            f"31–60: {r['days_31_60']:,.0f}",
            f"61–90: {r['days_61_90']:,.0f}",
            f"91–120: {r['days_91_120']:,.0f}",
            f"121–150: {r['days_121_150']:,.0f}",
            f"151–180: {r['days_151_180']:,.0f}",
            f"181–210: {r['days_181_210']:,.0f}",
            f">210: {r['above_210']:,.0f}"
        ])
        out += f"• {r['party_name']} → AED {r['balance']:,.2f}\n  Aging: {aging_str}\n\n"

    return out


# --------------------------------------------------
# 🚦 Intent Router
# --------------------------------------------------
# --------------------------------------------------
# 🚦 Main router
# --------------------------------------------------
# --------------------------------------------------
# 🚦 Main router  –  final, cleaned-up
# --------------------------------------------------
# --------------------------------------------------
# 🚦 Main router  – fully updated
# --------------------------------------------------
def handle_query_intent(question: str):
    """
    Detect the user’s intent and dispatch to the right handler.
    """
    q = question.lower().strip()
    t = tokens(q)                                    # ← NEW
    divisions = extract_divisions_from_question(q)

    # Optional greeting
    if is_greeting(q):
        return "👋 Hello! Ask me about sales, customers, suppliers, receivables, or payables."

    # ── requested list length ─────────────────────────────────────────
    m = re.search(r"(top|highest|most|bottom|lowest|least|worst)\s*(\d+)", q)
    top_n = int(m.group(2)) if m else 5

    # ── 1. PROFIT / MARGIN QUERIES ───────────────────────────────────
    profit_keys = {
        "margin", "profitable", "profit", "unit cost", "sale price",
        "high profitability", "low profitability",
        "highest margin", "lowest margin", "worst margin", "best margin",
        "loss making", "loss-making"
    }
    if profit_keys & t:
        sales_df    = load_data_from_index("sales_data_sample_index")
        purchase_df = load_data_from_index("purchase_data_sample_index")

        if {"high", "best", "highest"} & t:
            return handle_profitable_products(sales_df, purchase_df, q, top_n, asc=False)
        if {"low", "worst", "lowest", "loss"} & t:
            return handle_profitable_products(sales_df, purchase_df, q, top_n, asc=True)

        return handle_profitable_products(sales_df, purchase_df, q, top_n)

    # ── 2. ALL OTHER KEYWORD ROUTING ─────────────────────────────────
    index_name = guess_index_from_keywords(question) or get_index_from_question(question)

    if not index_name:
        return run_rag_fallback(question) or "🤔 Sorry, I couldn’t find data relevant to that question."

    df = load_data_from_index(index_name)
    print("✅ Loaded index:", index_name)
    if df.empty:
        return "❌ No data found in the index."

    # Direct handlers that don’t need extra checks
    if index_name == "pending_do_index":
        return handle_pending_do(df, q, top_n)
    if index_name == "ap_data_index":
        return handle_payables_with_aging(df, q, top_n)
    if index_name == "ar_data_index":
        return handle_receivables_with_aging(df, q, top_n)

    # ── basic cleaning for the remaining datasets ────────────────────
    df.columns = [c.lower().strip() for c in df.columns]
    if 'docdate'  in df.columns: df['docdate']  = pd.to_datetime(df['docdate'],  errors='coerce')
    if 'quantity' in df.columns: df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    if 'netamount'in df.columns: df['netamount']= pd.to_numeric(df['netamount'], errors='coerce')

    # Suppliers
    if {"supplier", "suppliers", "vendor", "vendors"} & t:
        return handle_top_suppliers(df, q, top_n)

    # Products
    if {"product", "item", "seller", "bestseller"} & t:
        return handle_sales_products(df, q, divisions, top_n)

    # Receivables (fallback inside sales index)
    if {"receivable", "receivables", "aging", "ageing", "ar", "overdue"} & t:
        return handle_receivables_with_aging(df, q, top_n)

    # Customers
    if {"customer", "client", "buyer"} & t:
        return handle_customers(df, q, divisions, top_n)

    # Salesmen
    if {"salesman", "salesmen", "salesperson"} & t:
        return handle_salesmen(df, q, divisions, top_n)

    # ── final fallback ───────────────────────────────────────────────
    return run_rag_fallback(question) or "🤔 I’m not sure how to answer that with the data I have."


