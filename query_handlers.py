import re
import pandas as pd
from datetime import datetime, timedelta
from rag_utils import get_index_from_question, load_data_from_index, run_rag_fallback

# --------------------------------------------------
# 📅 Constants
# --------------------------------------------------
MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12
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
def apply_division_filter(df: pd.DataFrame, divisions):
    """
    divisions : list[str] | str | None
    """
    if divisions in (None, [], ''):
        return df

    # 👉 if it’s a plain string, wrap it so .isin() gets a list
    if isinstance(divisions, str):
        # if that string already looks like an error message
        # just return it unchanged, letting the caller display it
        if divisions.startswith("❌"):
            return divisions
        divisions = [divisions]

    if 'division' not in df.columns:
        return df

    df['division'] = df['division'].astype(str).str.lower().str.strip()
    filtered = df[df['division'].isin(divisions)]

    if filtered.empty:
        available = sorted(df['division'].dropna().unique().tolist())
        return (
        f"⚠️ No pending DOs match the requested filters.\n"
        f"✅ Available months: {', '.join(sorted(df['docdate'].dt.to_period('M').astype(str).unique()))}\n"
        f"✅ Available divisions: {', '.join(sorted(df['division'].unique()))}"
    )
    return filtered


from datetime import datetime, timedelta
import pandas as pd
import re

# ----------------------------------------
DEFAULT_YEAR = 2024          # 🔧 change once, everything updates
# ----------------------------------------

def apply_date_filter(df, question, default_year: int = DEFAULT_YEAR):
    """Filter df by dates mentioned in *question*.
       All relative phrases are evaluated against a simulated 'today' (end of DEFAULT_YEAR).
    """
    if 'docdate' not in df.columns:
        return df

    

    df['docdate'] = pd.to_datetime(df['docdate'], errors='coerce')
    print("📆 Available months in data:", sorted(df['docdate'].dt.to_period("M").unique().astype(str)))
    q = question.lower()
    
    # 🔁 Simulate "today" as the end of DEFAULT_YEAR (e.g., 2024-06-30)
    today = datetime(default_year, 6, 30)  # Or whatever fixed "today" you want
    start_of_year = datetime(default_year, 1, 1)

    # ── NEW: “last week” / “last 7 days” ─────────────────────────
    if re.search(r"last\s+(?:7\s*days|week)", q):
        start = today - timedelta(days=7)
        return df[(df['docdate'] >= start) & (df['docdate'] <= today)]
    

    # ── NEW: “last week” / “last 7 days” ─────────────────────────
    if re.search(r"last\s+(?:7\s*days|month)", q):
        start = today - timedelta(days=30)
        return df[(df['docdate'] >= start) & (df['docdate'] <= today)]

    # ── relative ranges ─────────────────────────────────────
    if "last quarter" in q or "quarter" in q:
        start = today - timedelta(days=90)
        return df[(df['docdate'] >= start) & (df['docdate'] <= today)]

    elif "this month" in q:
        start = today.replace(day=1)
        return df[(df['docdate'] >= start) & (df['docdate'] <= today)]

    elif "last 6 months" in q:
        start = today - pd.DateOffset(months=6) + timedelta(days=1)
        print(f"[🗓️ Filter] last 6 months → {start.date()} to {today.date()}")
        return df[(df['docdate'] >= start) & (df['docdate'] <= today)]


    elif "last 3 months" in q:
        start = today - pd.DateOffset(months=3) + timedelta(days=1)
        print(f"[🗓️ Filter] last 3 months → {start.date()} to {today.date()}")
        return df[(df['docdate'] >= start) & (df['docdate'] <= today)]


    elif "last month" in q:
        start = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        end = start + pd.DateOffset(months=1) - timedelta(days=1)
        filtered = df[(df['docdate'] >= start) & (df['docdate'] <= end)]
        if filtered.empty:
            return pd.DataFrame()  # force proper empty handling
        return filtered

    # ── dynamic “last <n> months” ───────────────────────────
    m = re.search(r"last (\d+) months?", q)
    if m:
        months = int(m.group(1))
        start = today - pd.DateOffset(months=months) + timedelta(days=1)
        return df[(df['docdate'] >= start) & (df['docdate'] <= today)]

    # ── explicit month names (within default_year) ──────────
    for mon, idx in MONTH_MAP.items():
        if re.search(rf"\b{mon}\b", q):  # exact match
            return df[
                (df['docdate'].dt.month == idx) &
                (df['docdate'].dt.year == default_year)
            ]

    # ── fallback: entire default_year ───────────────────────
    return df[df['docdate'].dt.year == default_year]

# --------------------------------------------------
# 🧠 Intent helpers
# --------------------------------------------------
def tokens(q: str) -> set[str]:
    """Return lowercase word tokens from question."""
    return set(re.findall(r"\b\w+\b", q.lower()))

def is_greeting(q: str) -> bool:
    return {"hi", "hello", "hey"} & tokens(q)

VALID_DIVS = {
    "div_0097926054",
    "div_0556003793",
    "div_1434764897",
    "div_1742449716",
    "div_2006915470"
}

def extract_divisions_from_question(q: str) -> list[str] | str:
    """
    Extract and normalize division references from user query.
    Returns a list of valid divisions, or an error message if invalid ones are detected.
    """
    q = q.lower()
    raw_matches = re.findall(r"\b(?:division|div)[ _]?0*([0-9]+)\b", q)
    explicit_matches = re.findall(r"div_[0-9]{10}", q)

    # Normalize
    all_matches = [f"div_{m.zfill(10)}" for m in raw_matches] + explicit_matches

    # Separate valid and invalid
    valid = [d for d in all_matches if d in VALID_DIVS]
    invalid = [d for d in all_matches if d not in VALID_DIVS]

    if all_matches and not valid:
        return f"❌ No such division(s): {', '.join(all_matches)}.\n✅ Available: {', '.join(sorted(VALID_DIVS))}"
    
    return valid


def guess_index_from_keywords(question: str):
    t = tokens(question)
    q_lower = question.lower()

    if re.search(r"\b(pending|open)[ _]?do?s?\b", q_lower) \
    or "delivery order" in q_lower \
    or "do to invoice" in q_lower \
    or "pending to invoice" in q_lower \
    or "dos pending to invoice" in q_lower:
        return "pending_do_index"



    # ✅ Receivables: prioritize this first
    if {"receivable", "receivables", "overdue", "ar", "ageing"} & t or re.search(r"\bmonthly aging\b", q_lower):
        return "ar_data_index"

    if {"payable", "payables", "ap", "unpaid", "due", "outstanding", "aging", "ageing", "overdue"} & t:
        return "ap_data_index"

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

    # ✅ Ensure netamount and quantity are numeric
    for col in ["netamount", "quantity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    grouped = df.groupby(group_field)[["netamount", "quantity"]].sum().reset_index()


    if rev_flag and qty_flag:
        top_rev = grouped.sort_values("netamount", ascending=asc).head(top_n)
        top_qty = grouped.sort_values("quantity", ascending=asc).head(top_n)
        result += "💰 Top by Revenue:\n"
        for _, r in top_rev.iterrows():
            result += f"• {r[group_field]} → AED {float(r['netamount']):,.2f}, Qty {int(float(r['quantity']))}\n"
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
            result += f"• {r[group_field]} → AED {float(r['netamount']):,.2f}, Qty {int(float(r['quantity']))}\n"
        return result

# --------------------------------------------------
# 🎯 Main Handlers
# --------------------------------------------------
def handle_sales_products(df, question, divisions=None, top_n=5):
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    df = apply_division_filter(df, divisions)
    if isinstance(df, str): return df
    df = apply_date_filter(df, question)
    if df.empty: return "No product data available for the given filters."

    # 🩹 Default to both revenue and quantity if not mentioned
    if not any(k in question.lower() for k in ["revenue", "amount", "value", "quantity", "qty", "volume"]):
        question += " revenue and quantity"

    return format_dual_ranking(df, "stockdescription", "products", question, top_n, determine_sort_order(question))

def handle_customers(df, question, divisions=None, top_n=5):
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    df = apply_division_filter(df, divisions)
    if isinstance(df, str): return df
    df = apply_date_filter(df, question)
    if df.empty: return "No customer data available."

    # 🩹 Default to both revenue and quantity if not mentioned
    if not any(k in question.lower() for k in ["revenue", "amount", "value", "quantity", "qty", "volume"]):
        question += " revenue and quantity"

    return format_dual_ranking(df, "party_name", "customers", question, top_n, determine_sort_order(question))



def handle_salesmen(df, question, divisions=None, top_n=5):
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df = apply_division_filter(df, divisions)
    if isinstance(df, str): return df
    df = apply_date_filter(df, question)
    if df.empty: return "No salesman data available."

    # 🔧 Patch: If neither "revenue" nor "quantity" is mentioned, assume both
    if not any(k in question.lower() for k in ["revenue", "amount", "value", "quantity", "qty", "volume"]):
        question += " revenue and quantity"

    return format_dual_ranking(df, "salesman", "salesmen", question, top_n, determine_sort_order(question))


# --------------------------------------------------
# 💸  Simple sales-total handler
# --------------------------------------------------
# --------------------------------------------------
# 💸  Simple sales-total handler (revised)
# --------------------------------------------------
def handle_sales_summary(df: pd.DataFrame, question: str) -> str:
    """
    Return the grand-total Net Amount for the date / division span
    implied by *question* (e.g. “last week”, “Jan”, “last 3 months”).
    """
    df = df.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # 🔁 Normalize net_amount → netamount
    if "net_amount" in df.columns and "netamount" not in df.columns:
        df.rename(columns={"net_amount": "netamount"}, inplace=True)


    # ensure we have a numeric netamount column
    if "netamount" not in df.columns:
        if "amount" in df.columns:
            df["netamount"] = df["amount"]
        else:
            return "❌ Sales file lacks a ‘netamount’ column."

    df["netamount"] = pd.to_numeric(df["netamount"], errors="coerce").fillna(0)

    # optional division & date filters
    df = apply_division_filter(df, extract_divisions_from_question(question))
    if isinstance(df, str):                      # error string came back
        return df

    df = apply_date_filter(df, question)
    if df.empty:
        return "⚠️ No sales data for the requested period."

    total = df["netamount"].sum()
    return f"💰 Sales for the requested period: **AED {total:,.2f}**"

# --------------------------------------------------
# 📦  Simple purchase-total handler
# --------------------------------------------------
def handle_purchase_summary(df: pd.DataFrame, question: str) -> str:
    """
    Returns the total Net Amount from purchase data,
    filtered by division and/or time range from the question.
    """
    df = df.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # 🔁 Normalize net_amount → netamount
    if "net_amount" in df.columns and "netamount" not in df.columns:
        df.rename(columns={"net_amount": "netamount"}, inplace=True)


    # ensure we have a numeric netamount
    if "netamount" not in df.columns:
        if "amount" in df.columns:
            df["netamount"] = df["amount"]
        else:
            return "❌ Purchase file lacks a ‘netamount’ column."

    df["netamount"] = pd.to_numeric(df["netamount"], errors="coerce").fillna(0)

    df = apply_division_filter(df, extract_divisions_from_question(question))
    if isinstance(df, str):
        return df

    df = apply_date_filter(df, question)
    if df.empty:
        return "⚠️ No purchase data for the requested period."

    total = df["netamount"].sum()
    return f"🧾 Purchases for the requested period: **AED {total:,.2f}**"

# --------------------------------------------------
# 🔁  Compare Sales vs Purchases
# --------------------------------------------------
def handle_sales_vs_purchases(question: str) -> str:
    """
    Compares total sales and purchases for a given period.
    """
    sales_df = load_data_from_index("sales_data_sample_index")
    purchase_df = load_data_from_index("purchase_data_sample_index")

    for df in [sales_df, purchase_df]:
        # 1. Normalize column names
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # 2. Rename net_amount → netamount if needed
        if "net_amount" in df.columns and "netamount" not in df.columns:
            df.rename(columns={"net_amount": "netamount"}, inplace=True)

        # 3. Ensure netamount column exists
        if "netamount" not in df.columns:
            if "amount" in df.columns:
                df["netamount"] = df["amount"]
            else:
                return "❌ One of the files lacks a ‘netamount’ column."

        # 4. Ensure it's numeric
        df["netamount"] = pd.to_numeric(df["netamount"], errors="coerce").fillna(0)

    # Apply filters
    divisions = extract_divisions_from_question(question)
    sales_df = apply_division_filter(sales_df, divisions)
    purchase_df = apply_division_filter(purchase_df, divisions)

    sales_df = apply_date_filter(sales_df, question)
    purchase_df = apply_date_filter(purchase_df, question)

    if sales_df.empty and purchase_df.empty:
        return "⚠️ No sales or purchase data for the requested period."

    total_sales = sales_df["netamount"].sum()
    total_purchases = purchase_df["netamount"].sum()
    diff = total_sales - total_purchases
    trend = "surplus" if diff >= 0 else "deficit"

    return (
        f"📊 **Sales vs Purchases** for the requested period:\n\n"
        f"• 💰 Total Sales: **AED {total_sales:,.2f}**\n"
        f"• 🧾 Total Purchases: **AED {total_purchases:,.2f}**\n\n"
        f"➡️ Net {trend}: **AED {abs(diff):,.2f}**"
    )


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

    df_original = df.copy()  # Preserve full dataset for fallback

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

    divisions = extract_divisions_from_question(question)
    if isinstance(divisions, str):  # it's an error message
        return divisions


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

    # 🔎 Special case: user asked for cost > price (loss-making products)
    if re.search(r"(unit\s+cost\s*>\s*sale\s*price|cost\s*>\s*price)", question.lower()):
        filtered = merged[merged['avg_cost'] > merged['unit_sale_price']]
        if filtered.empty:
            return "✅ No products found where unit cost exceeds sale price."
        
        out = "🚨 Products where Unit Cost > Sale Price\n\n"
        for _, r in filtered.sort_values("margin_pct").head(top_n).iterrows():
            out += (
                f"• {r['stockdescription']} → "
                f"Margin {r['margin_pct']:.1f}%, "
                f"Profit AED {r['profit']:,.2f}, "
                f"Sales AED {r['total_sales']:,.2f}, "
                f"Unit Cost AED {r['avg_cost']:.2f}, "
                f"Unit Price AED {r['unit_sale_price']:.2f}\n"
            )
        return out


    

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
# 2️⃣  Pending-DO handler  –  entity-aware version
# -------------------------------------------------
# def handle_pending_do(df: pd.DataFrame, question: str, top_n: int = 10):
#     # ── normalise column names ─────────────────────────────────────
#     df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
#     if "date" in df.columns and "docdate" not in df.columns:
#         df = df.rename(columns={"date": "docdate"})
#     if "amount" not in df.columns:
#         return "❌ Column 'amount' missing in pending_do data."

#     # keep an untouched copy for the fallback
#     df_original = df.copy()

#     df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
#     if "salesman" in df.columns:
#         df["salesman"] = df["salesman"].astype(str).str.lower()
#     if "customer" in df.columns:
#         df["customer"] = df["customer"].astype(str).str.lower()

#     # ── extract entities ───────────────────────────────────────────
#     q_lower       = question.lower()
#     salesmen_ids  = extract_entities_from_question(q_lower, "salesman")
#     customers_ids = extract_entities_from_question(q_lower, "customer")

#     # ── decide grouping field ──────────────────────────────────────
#     if "salesman" in q_lower and "customer" in q_lower:
#         group_fields = ["salesman", "customer"]
#     elif "customer" in q_lower:
#         group_fields = ["customer"]
#     elif "salesman" in q_lower:
#         group_fields = ["salesman"]
#     elif salesmen_ids:
#         group_fields = ["customer"]
#     else:
#         group_fields = ["salesman"]

#     group_field = group_fields[-1]  # for fallback reference

#     # ── division & date filters ────────────────────────────────────
#     divisions = extract_divisions_from_question(question)
#     if isinstance(divisions, str):        # error string returned
#         return divisions

#     df = apply_division_filter(df, divisions)
#     if isinstance(df, str):
#         return df

#     df = apply_date_filter(df, question)

#     # ── entity filters ─────────────────────────────────────────────
#     if salesmen_ids:
#         ids = [s.replace("salesman_", "").replace("man_", "").lower() for s in salesmen_ids]
#         df = df[df["salesman"].isin(ids)]
#     if customers_ids:
#         ids = [c.replace("customer_", "").lower() for c in customers_ids]
#         df = df[df["customer"].isin(ids)]

#     # ── empty? → fallback to “this month” total ────────────────────
#     if df.empty:
#         available_divs = sorted(            # pull from the *original* dataframe
#             df_original.get("division", pd.Series()).dropna().unique().tolist()
#         )
#         suggest = (
#             "⚠️ No pending DOs match the requested filters.\n"
#             f"📌 Available divisions: {', '.join(available_divs) or 'None in file'}"
#         )
#         return suggest

#    # ── group & aggregate ──────────────────────────────────────────
#     grouped = (
#         df.groupby(group_fields, as_index=False)
#         .agg(total_pending=("amount", "sum"))
#         .sort_values("total_pending", ascending=False)
#     )

#     # ── format output ──────────────────────────────────────────────
#     header = f"📦 Pending DOs – grouped by {', '.join([f.title() for f in group_fields])}"
#     if divisions:
#         header += f" | Divisions: {', '.join(divisions)}"
#     if salesmen_ids:
#         header += f" | Salesman filter: {', '.join(salesmen_ids)}"
#     if customers_ids:
#         header += f" | Customer filter: {', '.join(customers_ids)}"

#     lines = []
#     for _, row in grouped.head(top_n).iterrows():
#         key_parts = [str(row[f]) for f in group_fields]
#         lines.append(f"• {' / '.join(key_parts)} → AED {row['total_pending']:,.2f}")

#     return header + "\n\n" + "\n".join(lines)

#     # ── format output ──────────────────────────────────────────────
#     header = f"📦 Pending DOs – grouped by {group_field.title()}"
#     if divisions:
#         header += f" | Divisions: {', '.join(divisions)}"
#     if salesmen_ids:
#         header += f" | Salesman filter: {', '.join(salesmen_ids)}"
#     if customers_ids:
#         header += f" | Customer filter: {', '.join(customers_ids)}"
#     lines = [
#         f"• {r[group_field]} → AED {r['total_pending']:,.2f}"
#         for _, r in grouped.iterrows()
#     ]
#     return header + "\n\n" + "\n".join(lines)
# ----------------------------------------------------------------------
# Pending-DO handler  – final patch
# ----------------------------------------------------------------------
def handle_pending_do(df: pd.DataFrame, user_input: str, top_n: int = 10) -> str:
    """
    Natural-language handler for Pending Delivery Orders.
    Handles:
      • salesman / customer filters (any length IDs, e.g. man_123)
      • division filters  (div_xxxxxxxxxx  or “Division 1” / “Division 2” mapping)
      • amount thresholds  (above / over AED n)
      • month / year phrases  +  “this month”
      • aging  (“over 90 days”)
      • grouped summaries  or  total fallback
    """

    import re
    from datetime import datetime

    # ------- normalise dataframe ------------------------------------
    now = pd.to_datetime("2025-06-30")                # simulated “today”
    df = df.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df['date']   = pd.to_datetime(df['date'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)

    for c in ('division', 'salesman', 'customer'):
        df[c] = df[c].astype(str).str.lower().str.strip()

    q = user_input.lower()

    # ------- division filter (code or plain “division 1/2/3…”) -------
    div_codes = re.findall(r"div_\d{10}", q)
    if "division 1" in q: div_codes.append("div_2006915470")
    if "division 2" in q: div_codes.append("div_0097926054")
    if "division 3" in q: div_codes.append("div_0556003793")
    if div_codes:
        df = df[df['division'].isin(div_codes)]

    # ------- salesman / customer ID filters --------------------------
    sal_ids  = re.findall(r"man_\d+", q)          # ← any length digits
    cust_ids = re.findall(r"cus_\d+", q)

    if sal_ids:
        df = df[df['salesman'].isin(sal_ids)]
    if cust_ids:
        df = df[df['customer'].isin(cust_ids)]

    # ------- amount threshold  (strict “> N”) ------------------------
    m_amt = re.search(r"(?:above|over)\s*(?:aed\s*)?([\d,]+)", q)
    if m_amt:
        threshold = float(m_amt.group(1).replace(",", ""))
        df = df[df['amount'] > threshold]

    # ------- explicit month / “this month” / year --------------------
    month_map = dict(jan=1,feb=2,mar=3,apr=4,may=5,jun=6,
                     jul=7,aug=8,sep=9,oct=10,nov=11,dec=12)
    for mon, idx in month_map.items():
        if rf"\b{mon}\b" in q:
            df = df[df['date'].dt.month == idx]

    if "this month" in q:
        df = df[(df['date'].dt.year == now.year) &
                (df['date'].dt.month == now.month)]

    year_hit = re.search(r"in\s*(20\d{2})", q)
    if year_hit:
        df = df[df['date'].dt.year == int(year_hit.group(1))]

    # ------- aging filter  (“over 60 days”) --------------------------
    m_age = re.search(r"over\s*(\d+)\s*days", q)
    if m_age:
        days = int(m_age.group(1))
        df = df[(now - df['date']).dt.days > days]

    if df.empty:
        return "⚠️ No pending DOs match the requested filters."

    # ------- grouping & special cases --------------------------------
    if "which" in q and "customer" in q:              # distinct list
        cust_list = ", ".join(sorted(df['customer'].unique()))
        return f"📋 Customers with open DOs:\n{cust_list}"

    if "by salesman" in q:
        grp = df.groupby('salesman')['amount'].sum().sort_values(ascending=False).head(top_n)
        return _fmt_summary("Salesman", grp)

    if "by customer" in q:
        grp = df.groupby('customer')['amount'].sum().sort_values(ascending=False).head(top_n)
        return _fmt_summary("Customer", grp)

    if "by division" in q:
        grp = df.groupby('division')['amount'].sum().sort_values(ascending=False).head(top_n)
        return _fmt_summary("Division", grp)

    if "top" in q:
        n_match = re.search(r"top\s*(\d+)", q)
        n = int(n_match.group(1)) if n_match else top_n
        grp = df.groupby('customer')['amount'].sum().sort_values(ascending=False).head(n)
        return _fmt_summary(f"Top {n} Customers", grp)

    # ------- default: total -----------------------------------------
    total = df['amount'].sum()
    return f"🔄 Total pending DO amount: AED {total:,.2f}"


# helper --------------------------------------------------------------
def _fmt_summary(title: str, series: pd.Series) -> str:
    out = [f"📦 Pending DOs – grouped by {title}"]
    for k, v in series.items():
        out.append(f"• {k} → AED {v:,.2f}")
    return "\n".join(out)



def handle_receivables_with_aging(df: pd.DataFrame, question: str, top_n: int = 10):
    """
    A flexible AR aging handler that can:
      • Show a specific party’s aging
      • Summarise by division
      • Produce a monthly aging roll-up
      • Filter customers whose oldest bucket exceeds an asked threshold
      • Default to the top-N parties by balance
    """
    # ── normalise ────────────────────────────────────────────────────
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df['balance'] = pd.to_numeric(df['balance'], errors='coerce')
    df.dropna(subset=['party_name', 'balance'], inplace=True)

    # ── optional division filter ────────────────────────────────────
    divisions = extract_divisions_from_question(question)
    if isinstance(divisions, str):  # it's an error message
        return divisions

    df = apply_division_filter(df, divisions)
    if isinstance(df, str): return df

    if df.empty: return "⚠️ No receivables data for the given filters."

    # ── aging buckets coercion ──────────────────────────────────────
    aging_cols = [
        'days_0_30', 'days_31_60', 'days_61_90', 'days_91_120',
        'days_121_150', 'days_151_180', 'days_181_210', 'days_above_210'
    ]
    for c in aging_cols:
        df[c] = pd.to_numeric(df.get(c, 0), errors='coerce').fillna(0)

    q = question.lower()

    # ────────────────────────────────────────────────────────────────
    # 1️⃣ PARTY-SPECIFIC AGING
    # ────────────────────────────────────────────────────────────────
    party_ids = extract_entities_from_question(q, "party")
    if party_ids:
        sub = df[df['party_name'].str.lower().isin(party_ids)]
        if sub.empty:
            return f"❌ No data found for: {', '.join(party_ids)}"

        out = f"📄 Receivables Aging – {' & '.join(party_ids)}\n\n"
        for _, r in sub.iterrows():
            aging_str = " | ".join([f"{c.replace('days_', '').replace('_', '–')}: {r[c]:,.0f}" for c in aging_cols])
            out += f"• {r['party_name']} → AED {r['balance']:,.2f}\n  Aging: {aging_str}\n\n"
        return out

    # ────────────────────────────────────────────────────────────────
    # 2️⃣ DIVISION-WISE SUMMARY
    # ────────────────────────────────────────────────────────────────
    if "division-wise" in q or "division-wise" in q or "by division" in q:
        if 'division' not in df.columns:
            return "❌ Division column not present in the AR data."

        grp = df.groupby('division', as_index=False)[['balance'] + aging_cols].sum()
        grp = grp.sort_values('balance', ascending=False)

        out = "🏢 Division-wise Receivables Aging Summary\n\n"
        for _, r in grp.iterrows():
            aging_str = " | ".join([f"{c.replace('days_', '').replace('_', '–')}: {r[c]:,.0f}" for c in aging_cols])
            out += f"• {r['division']} → AED {r['balance']:,.2f}\n  Aging: {aging_str}\n\n"
        return out

    # ────────────────────────────────────────────────────────────────
    # 3️⃣ MONTHLY AGING BREAKDOWN
    # ────────────────────────────────────────────────────────────────
    if "monthly" in q and 'docdate' in df.columns:
        df['docdate'] = pd.to_datetime(df['docdate'], errors='coerce')
        df = df[df['docdate'].dt.year == DEFAULT_YEAR]        # keep 2024 only
        df['month'] = df['docdate'].dt.month_name().str[:3]   # Jan, Feb, …
        grp = df.groupby('month')[['balance'] + aging_cols].sum().reset_index()

        # order calendar months Jan→Dec
        month_order = list(MONTH_MAP.keys())
        grp['order'] = grp['month'].str.lower().map(lambda m: month_order.index(m))
        grp = grp.sort_values('order')

        out = "📅 Monthly Aging Report (2024)\n\n"
        for _, r in grp.iterrows():
            aging_str = " | ".join([f"{c.replace('days_', '').replace('_', '–')}: {r[c]:,.0f}" for c in aging_cols])
            out += f"• {r['month'].capitalize()} → AED {r['balance']:,.2f}\n  Aging: {aging_str}\n\n"
        return out

    # ────────────────────────────────────────────────────────────────
    # 4️⃣ THRESHOLD FILTER  (e.g. "aging > 60 days")
    # ────────────────────────────────────────────────────────────────
    # 4️⃣ THRESHOLD FILTER  (e.g. "aging > 60 days", "overdue above 90", etc.)
    threshold_match = re.search(r"(?:aging|overdue)[^\d]{0,5}(\d{2,3})\s*days?", q)
    amt_match = re.search(r"(?:amount|balance|aed)?\s*above\s*(?:aed\s*)?([\d,]+)", q)

    if threshold_match:
        thresh = int(threshold_match.group(1))
        bucket_cols = [c for c in aging_cols if int(c.split('_')[-1]) > thresh]

        sub = df[df[bucket_cols].sum(axis=1) > 0]

        if amt_match:
            min_amt = float(amt_match.group(1).replace(",", ""))
            sub = sub[sub["balance"] > min_amt]

        sub = sub.sort_values("balance", ascending=False).head(top_n)

        out = f"🚩 Customers with Aging > {thresh} Days"
        if amt_match:
            out += f" and Balance > AED {min_amt:,.0f}"
        out += "\n\n"

        for _, r in sub.iterrows():
            overdue = int(r[bucket_cols].sum())
            out += f"• {r['party_name']} → AED {r['balance']:,.2f} (Overdue {overdue:,} in {', '.join(bucket_cols)})\n"
        return out


    # ────────────────────────────────────────────────────────────────
    # 5️⃣ DEFAULT: TOP-N PARTIES BY BALANCE
    # ────────────────────────────────────────────────────────────────
    grp = (
        df.groupby("party_name", as_index=False)[['balance'] + aging_cols]
          .sum()
          .sort_values("balance", ascending=False)
          .head(top_n)
    )

    out = f"🏦 Top {top_n} Customers with Receivables and Aging"
    if divisions: out += f" | Divisions: {', '.join(divisions)}"
    out += "\n\n"

    for _, r in grp.iterrows():
        aging_str = " | ".join([f"{c.replace('days_', '').replace('_', '–')}: {r[c]:,.0f}" for c in aging_cols])
        out += f"• {r['party_name']} → AED {r['balance']:,.2f}\n  Aging: {aging_str}\n\n"
    return out

def handle_payables_with_aging(df: pd.DataFrame, question: str, top_n: int = 10) -> str:
    """
    Flexible AP-aging handler (parity with AR):
      • Supplier-specific aging
      • Division-wise summary
      • Monthly roll-up
      • Threshold filters ( > 90 days  OR  > 6 months …)
      • Amount-based filters (above/below AED)
      • Default = top-N suppliers by (absolute) payable value
    """

    # ── normalize columns ───────────────────────────────
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce")
    df.dropna(subset=["party_name", "balance"], inplace=True)

    if 'docdate' in df.columns:
        df["docdate"] = pd.to_datetime(df["docdate"], errors="coerce")

    q = question.lower()

    # ── optional division + date filters ────────────────
    divisions = extract_divisions_from_question(q)
    df = apply_division_filter(df, divisions)
    if isinstance(df, str): return df
    if df.empty: return "⚠️ No payables data for the given filters."

    df = apply_date_filter(df, question)
    if df.empty: return "⚠️ No AP data for the selected time period."

    # ── aging buckets ───────────────────────────────────
    aging_cols = [
        "days_0_30", "days_31_60", "days_61_90", "days_91_120",
        "days_121_150", "days_151_180", "days_181_210", "above_210"
    ]
    for c in aging_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        else:
            df[c] = 0  # add missing aging column with 0s


    # ────────────────────────────────────────────────────
    # 1️⃣ PARTY-SPECIFIC
    # ────────────────────────────────────────────────────
    party_ids = extract_entities_from_question(q, "party") + \
                extract_entities_from_question(q, "supplier") + \
                extract_entities_from_question(q, "vendor")
    if party_ids:
        sub = df[df["party_name"].str.lower().isin(party_ids)]
        if sub.empty:
            return f"❌ No data found for: {', '.join(party_ids)}"

        out = f"📄 Payables Aging – {' & '.join(party_ids)}\n\n"
        for _, r in sub.iterrows():
            aging_str = " | ".join([f"{c.replace('days_', '').replace('_', '–')}: {r[c]:,.0f}" for c in aging_cols])
            out += f"• {r['party_name']} → AED {r['balance']:,.2f}\n  Aging: {aging_str}\n\n"
        return out
    
    if "salesman" in q and "salesman" in df.columns:
        # group by salesman and filter buckets
        bucket_cols = [c for c in aging_cols if int(re.findall(r"\d+", c)[0]) >= 210]
        grp = (
            df.groupby("salesman", as_index=False)[["balance"] + bucket_cols].sum()
            .query(f"{' + '.join(bucket_cols)} > 0")        # only >210-day balances
            .sort_values("balance", ascending=False)
        )
        if grp.empty:
            return "✅ No salesmen with aging > 210 days."

        out = "🚩 Salesmen with Aging > 210 Days\n\n"
        for _, r in grp.iterrows():
            overdue = int(r[bucket_cols].sum())
            out += f"• {r['salesman']} → AED {r['balance']:,.2f} (Overdue {overdue:,})\n"
        return out

    # ────────────────────────────────────────────────────
    # 2️⃣ DIVISION-WISE
    # ────────────────────────────────────────────────────
    if any(k in q for k in ["division-wise", "by division"]) and "division" in df.columns:
        grp = df.groupby("division", as_index=False)[["balance"] + aging_cols].sum().sort_values("balance")
        out = "🏢 Division-wise Payables Aging Summary\n\n"
        for _, r in grp.iterrows():
            aging_str = " | ".join([f"{c.replace('days_', '').replace('_', '–')}: {r[c]:,.0f}" for c in aging_cols])
            out += f"• {r['division']} → AED {r['balance']:,.2f}\n  Aging: {aging_str}\n\n"
        return out

    # ────────────────────────────────────────────────────
    # 3️⃣ MONTHLY BY DIVISION
    # ────────────────────────────────────────────────────
    if "monthly" in q and "docdate" in df.columns and "division" in df.columns:
        df = df[df["docdate"].dt.year == DEFAULT_YEAR]
        df["month"] = df["docdate"].dt.month_name().str[:3]
        grp = df.groupby(["division", "month"])[["balance"] + aging_cols].sum().reset_index()

        month_order = list(MONTH_MAP.keys())
        grp["order"] = grp["month"].str.lower().map(lambda m: month_order.index(m))
        grp = grp.sort_values(["division", "order"])

        out = "📅 Monthly Payables by Division (2024)\n\n"
        for div in grp["division"].unique():
            out += f"🔹 {div}:\n"
            sub = grp[grp["division"] == div]
            for _, r in sub.iterrows():
                aging_str = " | ".join([f"{c.replace('days_', '').replace('_', '–')}: {r[c]:,.0f}" for c in aging_cols])
                out += f"  • {r['month']} → AED {r['balance']:,.2f}\n    Aging: {aging_str}\n"
            out += "\n"
        return out

    # ────────────────────────────────────────────────────
    # 4️⃣ AGING THRESHOLD (days or months)
    # ────────────────────────────────────────────────────
    m = re.search(r"(?:above|over|>\s*)(\d{2,3})\s*days", q)
    thresh_days = int(m.group(1)) if m else None
    if not thresh_days:
        m = re.search(r"(?:above|over|>\s*)(\d{1,2})\s*months?", q)
        thresh_days = int(m.group(1)) * 30 if m else None

    if thresh_days:
        bucket_cols = [c for c in aging_cols if int(re.findall(r"\d+", c)[0]) >= thresh_days]
        sub = df[df[bucket_cols].sum(axis=1) > 0]
        if sub.empty:
            return f"✅ No suppliers unpaid for over {thresh_days} days."

        sub = sub.sort_values("balance")
        out = f"🚩 Suppliers unpaid for over {thresh_days} days\n\n"
        for _, r in sub.iterrows():
            overdue = int(r[bucket_cols].sum())
            out += f"• {r['party_name']} → AED {r['balance']:,.2f} (Overdue {overdue:,})\n"
        return out

    # ────────────────────────────────────────────────────
    # 5️⃣ AMOUNT THRESHOLD (above/below AED X)
    # ────────────────────────────────────────────────────
    m_above = re.search(r"above\s+(?:aed\s*)?([\d,]+)", q)
    m_below = re.search(r"below\s+(?:aed\s*)?([\d,]+)", q)
    if m_above or m_below:
        threshold = float((m_above or m_below).group(1).replace(",", ""))
        sub = df.copy()
        if m_above:
            sub = sub[sub["balance"] <= -threshold]
            label = f"💰 Suppliers with Outstanding Above AED {threshold:,.0f}"
        elif m_below:
            sub = sub[sub["balance"] >= -threshold]
            label = f"💸 Suppliers with Outstanding Below AED {threshold:,.0f}"

        if sub.empty:
            return f"✅ No suppliers matching the requested threshold."

        sub = sub.groupby("party_name", as_index=False)[["balance"] + aging_cols].sum()
        sub = sub.sort_values("balance")

        out = label + "\n\n"
        for _, r in sub.head(top_n).iterrows():
            aging_str = " | ".join([f"{c.replace('days_', '').replace('_', '–')}: {r[c]:,.0f}" for c in aging_cols])
            out += f"• {r['party_name']} → AED {r['balance']:,.2f}\n  Aging: {aging_str}\n\n"
        return out

    # ────────────────────────────────────────────────────
    # 6️⃣ DEFAULT: TOP-N SUPPLIERS
    # ────────────────────────────────────────────────────
    grp = df.groupby("party_name", as_index=False)[["balance"] + aging_cols].sum()
    grp = grp.sort_values("balance").head(top_n)

    out = f"🏦 Top {top_n} Suppliers with Payables and Aging"
    if divisions:
        out += f" | Divisions: {', '.join(divisions)}"
    out += "\n\n"
    for _, r in grp.iterrows():
        aging_str = " | ".join([f"{c.replace('days_', '').replace('_', '–')}: {r[c]:,.0f}" for c in aging_cols])
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
def handle_query_intent(question: str):
    """
    Detects the user’s intent and dispatches to the right handler.

    • “profit / margin / unit cost / sale price / high profitability / low profitability …”
        →  handle_profitable_products   (needs BOTH sales & purchase indices)
    • supplier, product, customer, salesman …  →  existing single-index handlers
    • otherwise                                →  run_rag_fallback (safe-guarded)
    """
    q = question.lower().strip()
    divisions = extract_divisions_from_question(q)

    # ── figure out requested list length ───────────────────────────────
    m = re.search(
        r"(top|highest|most)\s*(\d+)|"
        r"(bottom|lowest|least|worst)\s*(\d+)",
        q
    )
    try:
        m = re.search(r"(top|highest|most)\s*(\d+)|"
                    r"(bottom|lowest|least|worst)\s*(\d+)", q)
        top_n = int(next(g for g in m.groups() if g and g.isdigit()))
    except:
        top_n = 5

    try:
        # ───────────────── 1. PROFIT / MARGIN QUERIES ──────────────────
        profit_keys = [
            "margin", "profitable", "profit", "unit cost", "sale price",
            "high profitability", "low profitability",
            "highest margin", "lowest margin", "worst margin", "best margin",
            "loss making", "loss-making"
        ]

        if any(k in q for k in profit_keys):
            sales_df = load_data_from_index("sales_data_sample_index")
            purchase_df = load_data_from_index("purchase_data_sample_index")

            # Determine direction explicitly if possible
            if "high" in q or "best" in q or "highest" in q:
                return handle_profitable_products(sales_df, purchase_df, q, top_n, asc=False)
            if "low" in q or "worst" in q or "lowest" in q or "loss" in q:
                return handle_profitable_products(sales_df, purchase_df, q, top_n, asc=True)

            # Fallback to default sorting
            return handle_profitable_products(sales_df, purchase_df, q, top_n)
    # ------------------------------------------------------------------
    #  … leave everything above unchanged …
    # ------------------------------------------------------------------

        # ─────────────── 2.  ALL OTHER KEYWORD ROUTING ────────────────
        # obvious sales-keywords override
        index_name = guess_index_from_keywords(question) or get_index_from_question(question)


        if not index_name:
            return run_rag_fallback(question) or "🤔 Sorry, I couldn’t find data relevant to that question."

        df = load_data_from_index(index_name)
        print("✅ Loaded index:", index_name)
        if df.empty:
            return "❌ No data found in the index."
        
        if re.search(r"\b(sales?|sold)\b", q) and re.search(r"\b(purchases?|bought)\b", q):
            return handle_sales_vs_purchases(q)
        
        # ── simple “sales total” queries ──────────────────────────
        if index_name == "sales_data_sample_index" \
        and re.search(r"\bsales\b", q) \
        and not re.search(r"\b(product|item|stock|profit|margin|quantity|qty)\b", q):
            return handle_sales_summary(df, q)

        if index_name == "purchase_data_sample_index" \
        and re.search(r"\bpurchase(s)?\b", q) \
        and not re.search(r"\b(product|item|stock|supplier|cost|qty|quantity|price|profit|margin)\b", q):
            return handle_purchase_summary(df, q)

        # ✅ Regex-based routing for document type handlers
        if re.search(r"\b(pending|open)[ _]?do?s?\b|delivery[_ ]?order|do(?:s)? to invoice|do?s? pending to invoice", q):
            return handle_pending_do(df, q, top_n)

        if re.search(r"\b(payables?|unpaid|outstanding|aging|due|ap|overdue)\b", q) and re.search(r"\b(supplier|vendor)\b", q):
            return handle_payables_with_aging(df, q, top_n)

        if re.search(r"\b(receivables?|aging|overdue|outstanding|balance|ar)\b", q) \
        and re.search(r"\b(customer|client|party|salesman|salesmen|salesperson)\b", q):
            return handle_receivables_with_aging(df, q, top_n)


        # basic cleaning (applies only to non-pending-do datasets)
        if 'amount' in df.columns and 'netamount' not in df.columns:
            df['netamount'] = df['amount']

        if 'docdate'  in df.columns: df['docdate']  = pd.to_datetime(df['docdate'],  errors='coerce')
        for col in ['quantity', 'netamount']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if any(k in q for k in ["payables", "payable", "unpaid", "outstanding", "aging", "due", "ap", "overdue"]) and \
        any(k in q for k in ["supplier", "suppliers", "vendor", "vendors"]):
            return handle_payables_with_aging(df, q, top_n)

        # 1️⃣ Exact product / customer / salesman intent first
        if any(k in q for k in ["supplier", "suppliers", "vendor", "vendors"]):
            return handle_top_suppliers(df, q, top_n)

        if any(k in q for k in ["product", "item", "stock", "bestseller"]):
            return handle_sales_products(df, q, divisions, top_n)

        if any(k in q for k in ["customer", "client", "buyer"]):
            return handle_customers(df, q, divisions, top_n)

        if any(k in q for k in ["salesman", "salesmen", "salesperson"]):
            return handle_salesmen(df, q, divisions, top_n)
        
        

        # 2️⃣ AR aging queries (run only if the above did not match)
        if any(k in q for k in [
            "receivable", "receivables", "aging", "ageing", "ar",
            "overdue", "outstanding", "balance", "due"
        ]):
            return handle_receivables_with_aging(df, q, top_n)


        # fallback
        return run_rag_fallback(question) or "🤔 I’m not sure how to answer that with the data I have."


        

        # default fallback
        return run_rag_fallback(question) or \
               "🤔 I’m not sure how to answer that with the data I have."

    except Exception as e:
        return f"❌ Failed to process query: {e}"
