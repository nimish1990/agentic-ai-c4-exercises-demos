import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Fix 1: Use absolute paths so the script works regardless of the working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create an SQLite database
db_engine = create_engine(f"sqlite:///{os.path.join(BASE_DIR, 'munder_difflin.db')}")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv(os.path.join(BASE_DIR, "quote_requests.csv"))
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv(os.path.join(BASE_DIR, "quotes.csv"))
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    # Use LOWER() on both sides so 'A4 Glossy Paper' correctly finds 'Glossy paper' seed rows.
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE LOWER(item_name) = LOWER(:item_name)
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################

import threading as _threading

# ---------------------------------------------------------------------------
# Transaction serialization (no helper function modified)
# ---------------------------------------------------------------------------
# smolagents executes multiple tool calls in parallel within a single step.
# SQLite's last_insert_rowid() is connection-scoped: if two INSERTs happen
# concurrently, each SELECT may return the other call's rowid → duplicate IDs.
# Re-binding create_transaction through a lock serialises INSERT+SELECT pairs
# so each call gets its own unique ID. The original helper is never edited.
_txn_lock = _threading.Lock()
_create_transaction_original = create_transaction

def create_transaction(item_name, transaction_type, quantity, price, date):
    """Thread-safe wrapper — serialises INSERT + last_insert_rowid() SELECT."""
    with _txn_lock:
        return _create_transaction_original(item_name, transaction_type, quantity, price, date)

import os
import dotenv
from smolagents import CodeAgent, ToolCallingAgent, tool, LiteLLMModel

# ---------------------------------------------------------------------------
# 1. Environment setup & model
# ---------------------------------------------------------------------------
dotenv.load_dotenv()

api_key = os.getenv("UDACITY_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

# Use LiteLLM so we can point at the Vocareum proxy
model = LiteLLMModel(
    model_id="openai/gpt-4o-mini",
    api_key=api_key,
    api_base=base_url,
)

# ---------------------------------------------------------------------------
# 2. Tools
# ---------------------------------------------------------------------------

# ── Catalog tools ────────────────────────────────────────────────────────────

# Mapping of common non-canonical name fragments → exact catalog name
# Used by normalize_item_name_tool and the orchestrator instructions
_ITEM_NAME_ALIASES: List[tuple] = [
    # glossy
    ("glossy",          "Glossy paper"),
    # matte
    ("matte",           "Matte paper"),
    # cardstock / card stock
    ("cardstock",       "Cardstock"),
    ("card stock",      "Cardstock"),
    # colorful/colourful + another paper type → resolve to THAT paper type (adjective only)
    # These must come BEFORE the generic "colored paper" entries so the longer alias wins
    ("colorful poster paper",      "Poster paper"),
    ("colourful poster paper",     "Poster paper"),
    ("colorful construction paper","Construction paper"),
    ("colourful construction paper","Construction paper"),
    ("colorful cardstock",         "Cardstock"),
    ("colourful cardstock",        "Cardstock"),
    # colored / colourful paper (standalone → Colored paper)
    ("colored paper",   "Colored paper"),
    ("coloured paper",  "Colored paper"),
    ("colourful paper", "Colored paper"),
    # construction paper
    ("construction paper", "Construction paper"),
    # recycled paper
    ("recycled paper",  "Recycled paper"),
    # standard / printer / copy paper
    ("standard copy paper",   "Standard copy paper"),
    ("standard printer paper","Standard copy paper"),
    ("white printer paper",   "Standard copy paper"),
    ("printer paper",         "Standard copy paper"),
    ("copy paper",            "Standard copy paper"),
    # A4 paper
    ("a4 paper",        "A4 paper"),
    ("a4 white",        "A4 paper"),
    ("a3 paper",        "A4 paper"),   # closest match; no A3 in catalog
    # washi / decorative adhesive tape
    ("washi tape",      "Decorative adhesive tape (washi tape)"),
    ("decorative tape", "Decorative adhesive tape (washi tape)"),
    # poster paper
    ("poster paper",    "Poster paper"),
    ("poster board",    "Large poster paper (24x36 inches)"),
    ("large poster",    "Large poster paper (24x36 inches)"),
    # eco / eco-friendly
    ("eco-friendly",    "Eco-friendly paper"),
    ("eco friendly",    "Eco-friendly paper"),
    # kraft
    ("kraft",           "Kraft paper"),
    # heavyweight
    ("heavyweight",     "Heavyweight paper"),
    ("heavy paper",     "Heavyweight paper"),
    # banner
    ("banner paper",    "Banner paper"),
    ("rolls of banner", "Rolls of banner paper (36-inch width)"),
    # letter-sized
    ("letter-sized",    "Letter-sized paper"),
    ("letter sized",    "Letter-sized paper"),
    # legal-size
    ("legal-size",      "Legal-size paper"),
    ("legal size",      "Legal-size paper"),
    # photo
    ("photo paper",     "Photo paper"),
    # paper cups / disposable cups
    ("paper cups",      "Paper cups"),
    ("disposable cups", "Disposable cups"),
    # paper napkins
    ("paper napkins",   "Paper napkins"),
    ("table napkins",   "Paper napkins"),
    # paper plates
    ("paper plates",    "Paper plates"),
    # envelopes
    ("envelopes",       "Envelopes"),
]


def _resolve_item_name(raw_name: str) -> str:
    """Return the closest official catalog name for a raw item description."""
    low = raw_name.lower().strip()
    # Exact match first
    for item in paper_supplies:
        canonical: str = str(item["item_name"])
        if canonical.lower() == low:
            return canonical
    # Alias table match (longest alias wins)
    best_match: str = ""
    best_len: int = 0
    for alias, canonical_alias in _ITEM_NAME_ALIASES:
        if alias in low and len(alias) > best_len:
            best_match = canonical_alias
            best_len = len(alias)
    return best_match if best_match else raw_name  # return original if nothing matched


@tool
def get_catalog_items_tool() -> str:
    """
    Get the full list of official paper items and products available in the Munder Difflin catalog.
    
    Use this tool to find the exact official name of an item to ensure there are no
    mismatches when checking inventory, generating quotes, or fulfilling orders.
    
    Returns:
        A comma-separated list of all official catalog item names.
    """
    item_names = [item["item_name"] for item in paper_supplies]
    return "Official Catalog Items: " + ", ".join(item_names)


@tool
def normalize_item_name_tool(item_name: str) -> str:
    """
    Resolve a customer-facing item description to the closest official catalog name.

    Use this tool whenever you receive an item name that might be a variant, informal
    description, or qualified version of a catalog item (e.g. 'Cardstock (white)',
    'A4 glossy paper', 'colored paper (assorted colors)'). It returns the exact
    catalog name you should use in all subsequent tool calls.

    Args:
        item_name: Any item description (can be informal or contain qualifiers).

    Returns:
        The closest matching official catalog item name, or the original name if no
        match is found (which means the item is genuinely not in the catalog).
    """
    resolved = _resolve_item_name(item_name)
    # Verify the resolved name is actually in the catalog
    for item in paper_supplies:
        if item["item_name"] == resolved:
            return f"Official catalog name: '{resolved}'  (mapped from '{item_name}')"
    return (
        f"Could not map '{item_name}' to any official catalog item. "
        f"Check get_catalog_items_tool for the full list."
    )


@tool
def get_catalog_item_price_tool(item_name: str) -> str:
    """
    Get the official base unit price for a specific catalog item.
    
    Use this tool before generating quotes to know the true base price of an item 
    before applying any bulk discounts.
    
    Args:
        item_name: The exact name of the official catalog item.
        
    Returns:
        The official unit price of the item, or an error if the item is not found.
        If the item is not found, try normalize_item_name_tool first to get the exact name.
    """
    for item in paper_supplies:
        if item["item_name"].lower() == item_name.lower():
            return f"The official base unit price for '{item['item_name']}' is ${item['unit_price']:.2f}."
    # Try auto-resolving via alias table before giving up
    resolved = _resolve_item_name(item_name)
    if resolved != item_name:
        for item in paper_supplies:
            if item["item_name"] == resolved:
                return (
                    f"Note: '{item_name}' was auto-mapped to official name '{resolved}'. "
                    f"The official base unit price for '{resolved}' is ${item['unit_price']:.2f}. "
                    f"Please use '{resolved}' as the item name in all subsequent tool calls."
                )
    return f"Item '{item_name}' was not found in the official catalog. Use normalize_item_name_tool to find the closest match."


# ── Inventory tools ─────────────────────────────────────────────────────────

@tool
def check_inventory_tool(item_name: str, request_date: str) -> str:
    """
    Check the current stock level for a specific paper item as of a given date,
    and flag whether a reorder is required based on the item's minimum stock threshold.

    Use this tool to find out how many units of a particular item are
    available on the requested date, so you can decide whether to fulfil
    an order from existing stock or arrange a supplier reorder.

    IMPORTANT: If the output says '⚠️ REORDER REQUIRED', the item MUST be
    restocked by calling reorder_stock_tool, even if the customer order can
    still be partially fulfilled.

    Args:
        item_name: The exact name of the paper item (e.g. 'A4 paper').
        request_date: ISO-format date string (YYYY-MM-DD) for the stock snapshot.

    Returns:
        A plain-text summary of the available stock, minimum threshold, and
        whether a reorder is needed.
    """
    df = get_stock_level(item_name, request_date)
    stock = int(df["current_stock"].iloc[0]) if not df.empty else 0

    # Look up the minimum stock threshold from the inventory table
    try:
        inv_df = pd.read_sql(
            "SELECT min_stock_level FROM inventory WHERE LOWER(item_name) = LOWER(:name)",
            db_engine,
            params={"name": item_name},
        )
        min_level = int(inv_df["min_stock_level"].iloc[0]) if not inv_df.empty else 0
    except Exception:
        min_level = 0

    needs_reorder = stock <= min_level

    status = (
        f"⚠️  REORDER REQUIRED — stock ({stock} units) is at or below the "
        f"minimum threshold ({min_level} units). "
        f"smart_order_tool will automatically handle the reorder for '{item_name}'."
        if needs_reorder
        else f"✅ Stock OK — {stock} units available (minimum threshold: {min_level} units)."
    )

    return (
        f"Stock level for '{item_name}' as of {request_date}: {stock} units available. "
        f"Minimum reorder threshold: {min_level} units. {status}"
    )



@tool
def check_all_inventory_tool(request_date: str) -> str:
    """
    Retrieve a summary of ALL items currently in stock as of the given date.

    Use this when you need a full view of available inventory to answer
    broad availability questions or to identify items to reorder.

    Args:
        request_date: ISO-format date string (YYYY-MM-DD).

    Returns:
        A formatted text list of every item with a positive stock level.
    """
    inventory: Dict[str, int] = get_all_inventory(request_date)
    if not inventory:
        return "No items currently in stock."
    lines = [f"  • {name}: {qty} units" for name, qty in inventory.items()]
    return "Available inventory as of {}:\n{}".format(request_date, "\n".join(lines))


@tool
def get_delivery_date_tool(request_date: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date for a given order quantity.

    Lead times are:
      - ≤10 units   → same day
      - 11–100     → 1 day
      - 101–1 000  → 4 days
      - >1 000     → 7 days

    Args:
        request_date: The starting date in ISO format (YYYY-MM-DD).
        quantity: Number of units to be ordered from the supplier.

    Returns:
        A plain-text statement of the estimated delivery date.
    """
    delivery = get_supplier_delivery_date(request_date, quantity)
    return (
        f"For a supplier order of {quantity} units placed on {request_date}, "
        f"estimated delivery date is {delivery}."
    )


# ── Financial tools ──────────────────────────────────────────────────────────

@tool
def cash_balance_tool(as_of_date: str) -> str:
    """
    Check the company's available cash balance as of a specific date.

    Use this tool to verify available funds before making large supplier orders,
    to ensure the company has sufficient cash to cover the cost of a reorder.
    If the cash balance is too low, flag the risk before proceeding with a
    large stock purchase from the supplier.

    Args:
        as_of_date: ISO format date string (YYYY-MM-DD) for which the cash
                    balance should be calculated.

    Returns:
        A formatted message showing the current cash balance as of the given date.
    """
    balance = get_cash_balance(as_of_date)
    return (
        f"Cash balance as of {as_of_date}: ${balance:,.2f}. "
        f"{'Sufficient funds available for supplier orders.' if balance > 500 else 'WARNING: Low cash balance — verify order cost before proceeding.'}"
    )


# ── Quoting tools ────────────────────────────────────────────────────────────

@tool
def search_quotes_tool(keywords: str, limit: int = 5) -> str:
    """
    Search the historical quote database using comma-separated keywords.

    Use this tool to find past quotes that are similar to a new customer
    request, so you can provide consistent and competitive pricing.

    Args:
        keywords: Comma-separated search terms (e.g. 'A4 paper, bulk, meeting').
        limit: Maximum number of historical quotes to return (default 5).

    Returns:
        A formatted summary of matching historical quotes.
    """
    terms = [k.strip() for k in keywords.split(",") if k.strip()]
    results = search_quote_history(terms, limit=limit)
    if not results:
        return "No matching historical quotes found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. Total: ${r['total_amount']:.2f} | "
            f"Job: {r.get('job_type','')} | "
            f"Size: {r.get('order_size','')} | "
            f"Event: {r.get('event_type','')} | "
            f"Explanation: {r.get('quote_explanation','')[:120]}..."
        )
    return "Historical quotes matching '{}':\n{}".format(keywords, "\n".join(lines))


# ── Order-fulfillment tools ──────────────────────────────────────────────────

@tool
def fulfill_order_tool(
    item_name: str,
    quantity: int,
    unit_price: float,
    order_date: str,
    apply_bulk_discount: bool = False,
) -> str:
    """
    Finalise a customer sale by recording a 'sales' transaction in the database.

    IMPORTANT: This tool validates available stock before recording the sale.
    If stock is insufficient it will return an error — do NOT call this tool
    if you have not confirmed enough stock is available. Use reorder_stock_tool
    for items that are out of stock or below the requested quantity.

    Optionally apply a bulk discount (≥500 units → 10 % off) for SELL items only.
    Never pass apply_bulk_discount=True for reorder items.

    Args:
        item_name: Exact name of the paper item being sold.
        quantity: Number of units sold.
        unit_price: Per-unit price BEFORE any discount.
        order_date: ISO-format date string (YYYY-MM-DD) of the sale.
        apply_bulk_discount: Whether to apply a 10 % bulk discount (SELL items only).

    Returns:
        Confirmation message with transaction ID and final price, or an error if
        insufficient stock is available.
    """
    # --- Stock validation guard ---
    stock_df = get_stock_level(item_name, order_date)
    available = int(stock_df["current_stock"].iloc[0]) if not stock_df.empty else 0
    if available < quantity:
        return (
            f"ERROR: Cannot fulfill order for '{item_name}' — only {available} units "
            f"available as of {order_date}, but {quantity} were requested. "
            f"No transaction has been recorded. "
            f"Please call reorder_stock_tool to place a supplier order for this item instead."
        )

    price = unit_price * quantity
    if apply_bulk_discount and quantity >= 500:
        price *= 0.90  # 10 % bulk discount
    txn_id = create_transaction(item_name, "sales", quantity, price, order_date)
    discount_note = " (10% bulk discount applied)" if apply_bulk_discount and quantity >= 500 else ""
    return (
        f"Order fulfilled! Sold {quantity} units of '{item_name}' on {order_date} "
        f"for ${price:.2f}{discount_note}. Transaction ID: {txn_id}."
    )


@tool
def reorder_stock_tool(
    item_name: str,
    quantity: int,
    unit_price: float,
    order_date: str,
    customer_sale_price: float = 0.0,
    delivery_date: str = "",
) -> str:
    """
    Place a stock replenishment order with the supplier AND optionally record
    the corresponding customer sale on the delivery date.

    This records a 'stock_orders' transaction (supplier purchase) in the database.
    If customer_sale_price > 0 and delivery_date is provided, it also records a
    'sales' transaction on the delivery date at the customer price (with 10% bulk
    discount automatically applied if quantity >= 500).

    Args:
        item_name: Exact name of the paper item to restock.
        quantity: Number of units to order from the supplier.
        unit_price: Per-unit cost charged by the supplier (always full price, no discount).
        order_date: ISO-format date string (YYYY-MM-DD) of the supplier order.
        customer_sale_price: Per-unit price charged to the customer (optional).
                             When > 0, a customer sale is also recorded on delivery_date.
                             Set to 0 if this reorder is purely for stock replenishment
                             (not tied to a specific customer order).
        delivery_date: ISO-format date string (YYYY-MM-DD) when stock arrives from supplier.
                       Required when customer_sale_price > 0.

    Returns:
        Confirmation message with supplier transaction ID, delivery date,
        and (if applicable) customer sale transaction ID and revenue.
    """
    # --- Code-level catalog guard ---
    catalog_entry = next(
        (item for item in paper_supplies if item["item_name"].lower() == item_name.lower()),
        None
    )
    if catalog_entry is None:
        return (
            f"ERROR: Cannot reorder '{item_name}' — this item is NOT in The Beaver's Choice "
            f"official catalog. No transaction has been recorded. Please use an exact catalog item name."
        )

    # Always use the official catalog price for the supplier cost — no discounts
    official_price: float = float(catalog_entry["unit_price"])
    safe_unit_price: float = (
        official_price
        if (unit_price <= 0 or abs(unit_price - official_price) > official_price * 0.5)
        else unit_price
    )

    # --- Record supplier purchase (stock_orders) ---
    supplier_cost = safe_unit_price * quantity
    txn_id = create_transaction(item_name, "stock_orders", quantity, supplier_cost, order_date)
    effective_delivery = get_supplier_delivery_date(order_date, quantity)

    result = (
        f"Reorder placed! Ordered {quantity} units of '{item_name}' on {order_date} "
        f"for ${supplier_cost:.2f} (unit cost ${safe_unit_price:.3f}). "
        f"Estimated delivery: {effective_delivery}. Supplier Transaction ID: {txn_id}."
    )

    # --- Optionally record customer sale (sales) on delivery date ---
    if customer_sale_price > 0:
        # Validate delivery_date — reject hallucinated years outside 2025-2030
        def _is_valid_date(d: str) -> bool:
            try:
                from datetime import date as _date
                parsed = _date.fromisoformat(d)
                return 2025 <= parsed.year <= 2030
            except Exception:
                return False

        if delivery_date and _is_valid_date(delivery_date):
            sale_date = delivery_date
        else:
            # Fall back to computed delivery (ignores any hallucinated date)
            sale_date = effective_delivery
        # Apply 10% bulk discount to customer price if quantity >= 500
        discount_factor = 0.9 if quantity >= 500 else 1.0
        customer_unit = customer_sale_price * discount_factor
        customer_revenue = customer_unit * quantity
        sale_txn_id = create_transaction(item_name, "sales", quantity, customer_revenue, sale_date)
        discount_note = " (10% bulk discount applied)" if quantity >= 500 else ""
        result += (
            f"\nCustomer sale recorded! Sold {quantity} units of '{item_name}' on {sale_date} "
            f"for ${customer_revenue:.2f} (${customer_unit:.3f}/unit{discount_note}). "
            f"Customer Sale Transaction ID: {sale_txn_id}."
        )

    return result



@tool
def smart_order_tool(
    item_name: str,
    quantity: int,
    unit_price: float,
    order_date: str,
    customer_sale_price: float = 0.0,
    delivery_date: str = "",
) -> str:
    """
    Automatically decides SELL vs REORDER in Python code — not left to the LLM.

    Simple rule:
      SELL  (fulfill from stock NOW):
            available stock >= requested quantity  AND  stock > min_stock_level
      REORDER (place supplier order):
            available stock < requested quantity   OR   stock <= min_stock_level

    For SELL   : records a 'sales' transaction on order_date.
                 Applies 10% bulk discount automatically if quantity >= 500.
    For REORDER: records a 'stock_orders' transaction on order_date (supplier cost),
                 then records a 'sales' transaction on order_date at customer_sale_price
                 (same-day model — both legs hit the DB on the order date so inventory
                 stays neutral and cash immediately reflects the net margin).
                 10% bulk discount applied automatically if quantity >= 500.
                 If customer_sale_price is 0 or omitted, falls back to unit_price
                 (catalog price) so a customer sale is ALWAYS recorded on reorders.

    Args:
        item_name           : Exact catalog name of the item.
        quantity            : Number of units the customer wants.
        unit_price          : Catalog unit price.
        order_date          : ISO date (YYYY-MM-DD) of the customer request.
        customer_sale_price : Per-unit price to charge the customer for REORDER items.
                              Defaults to unit_price when omitted or 0 — a customer
                              sale is always recorded, even if not explicitly provided.
        delivery_date       : Expected delivery date for REORDER (ISO YYYY-MM-DD).
                              Informational only — the sale is recorded on order_date.

    Returns:
        Confirmation message with decision made, transaction IDs, and amounts.
    """
    # 0. Auto-resolve item_name to the official catalog name.
    #    This is a safety net in case the LLM skipped normalize_item_name_tool
    #    or passed a partially-qualified name (e.g. 'A4 Glossy Paper').
    resolved = next(
        (item["item_name"] for item in paper_supplies
         if item["item_name"].lower() == item_name.strip().lower()),
        None,
    )
    if resolved is None:
        # Try the alias table as a second pass
        from_alias = _resolve_item_name(item_name)
        resolved = next(
            (item["item_name"] for item in paper_supplies
             if item["item_name"].lower() == from_alias.lower()),
            None,
        )
    if resolved is None:
        return (
            f"ERROR: '{item_name}' could not be resolved to any official catalog item. "
            "No transaction recorded. Check get_catalog_items_tool for valid names."
        )
    item_name = resolved   # use the exact catalog name for all subsequent logic

    # 1. Current stock as of order_date
    stock_df = get_stock_level(item_name, order_date)
    stock = int(stock_df["current_stock"].iloc[0]) if not stock_df.empty else 0

    # 2. Minimum stock threshold from inventory table
    try:
        inv_df = pd.read_sql(
            "SELECT min_stock_level FROM inventory WHERE LOWER(item_name) = LOWER(:name)",
            db_engine,
            params={"name": item_name},
        )
        min_level = int(inv_df["min_stock_level"].iloc[0]) if not inv_df.empty else 0
    except Exception:
        min_level = 0

    # 3. Deterministic SELL vs REORDER decision
    can_sell = (stock >= quantity) and (stock > min_level)

    if can_sell:
        # ── SELL: fulfill from existing stock ──────────────────────────────
        price = unit_price * quantity
        if quantity >= 500:
            price *= 0.90          # 10% bulk discount
        txn_id = create_transaction(item_name, "sales", quantity, price, order_date)
        discount_note = " (10% bulk discount applied)" if quantity >= 500 else ""
        return (
            f"[SELL] Sold {quantity} units of '{item_name}' on {order_date} "
            f"for ${price:.2f}{discount_note}. Transaction ID: {txn_id}. "
            f"Stock was {stock} units (min threshold: {min_level})."
        )
    else:
        # ── REORDER: place supplier order ──────────────────────────────────
        catalog_entry = next(
            (item for item in paper_supplies
             if item["item_name"].lower() == item_name.lower()),
            None,
        )
        if catalog_entry is None:
            return (
                f"ERROR: '{item_name}' is not in the catalog. No transaction recorded."
            )
        official_price = float(catalog_entry["unit_price"])
        safe_unit_price = (
            official_price
            if (unit_price <= 0 or abs(unit_price - official_price) > official_price * 0.5)
            else unit_price
        )
        supplier_cost = safe_unit_price * quantity
        txn_id = create_transaction(item_name, "stock_orders", quantity, supplier_cost, order_date)
        effective_delivery = get_supplier_delivery_date(order_date, quantity)

        reason = (
            f"insufficient stock ({stock} < {quantity} requested)"
            if stock < quantity
            else f"stock ({stock}) at/below min threshold ({min_level})"
        )
        result = (
            f"[REORDER] Ordered {quantity} units of '{item_name}' on {order_date} "
            f"for ${supplier_cost:.2f} (${safe_unit_price:.3f}/unit). "
            f"Reason: {reason}. "
            f"Estimated delivery: {effective_delivery}. Supplier Transaction ID: {txn_id}."
        )

        # ── Two distinct reorder triggers → different accounting treatments ──
        #
        # Case A — INSUFFICIENT STOCK  (stock < quantity)
        #   Customer placed an order we cannot fill from shelf stock.
        #   Buy from supplier AND record the customer sale immediately
        #   (same-day model). Inventory stays neutral; cash shows net margin.
        #   If LLM omitted customer_sale_price, fall back to catalog price so
        #   revenue is never silently omitted.
        #
        # Case B — THRESHOLD REORDER  (stock <= min_level but stock >= quantity)
        #   Stock dipped to the safety floor. Pure inventory replenishment —
        #   no specific customer order is driving this. Only the supplier
        #   purchase (stock_orders) hits the books; NO customer sale recorded.
        if stock < quantity:
            # Case A: customer-order-driven → record customer sale
            effective_customer_price = customer_sale_price if customer_sale_price > 0 else safe_unit_price
            sale_date = order_date   # same-day model
            discount_factor = 0.9 if quantity >= 500 else 1.0
            customer_unit    = effective_customer_price * discount_factor
            customer_revenue = customer_unit * quantity
            sale_txn_id = create_transaction(
                item_name, "sales", quantity, customer_revenue, sale_date
            )
            discount_note = " (10% bulk discount applied)" if quantity >= 500 else ""
            fallback_note = " (customer price defaulted to catalog price)" if customer_sale_price <= 0 else ""
            result += (
                f"\nCustomer sale recorded: {quantity} units of '{item_name}' on {sale_date} "
                f"at ${customer_unit:.3f}/unit = ${customer_revenue:.2f}{discount_note}{fallback_note}. "
                f"Customer Sale Transaction ID: {sale_txn_id}."
                f" (Supplier delivery by {effective_delivery}.)"
            )
        else:
            # Case B: threshold-only restocking → NO customer sale
            result += (
                f"\nNote: Stock replenishment only — no customer sale recorded. "
                f"Stock ({stock} units) was at/below minimum threshold ({min_level} units). "
                f"Inventory will be replenished upon supplier delivery ({effective_delivery})."
            )

        return result



# ---------------------------------------------------------------------------
# 3. Worker Agents
# ---------------------------------------------------------------------------

inventory_agent = ToolCallingAgent(
    tools=[check_inventory_tool, check_all_inventory_tool, get_delivery_date_tool],
    model=model,
    name="inventory_agent",
    description=(
        "Handles all inventory-related questions for The Beaver's Choice Paper Company. "
        "Use this agent to: check stock levels for specific items, list all available "
        "inventory, and estimate supplier delivery dates for reorders. "
        "Always pass the exact item name and the request date (YYYY-MM-DD). "
        "CRITICAL: check_inventory_tool will tell you if '\u26a0\ufe0f REORDER REQUIRED' for each item. "
        "You MUST include this flag clearly in your final answer for each item, "
        "so the orchestrator knows which items need a supplier reorder. "
        "CRITICAL: Always conclude by returning a string with your final answer using the final_answer variable or function."
    ),
    max_steps=5,
)

quoting_agent = ToolCallingAgent(
    tools=[search_quotes_tool],
    model=model,
    name="quoting_agent",
    description=(
        "Generates competitive price quotes for The Beaver's Choice customers. "
        "The orchestrator provides you with exact item names, quantities, and verified unit prices in the task string, "
        "split into SELL items (fulfilled from existing stock) and REORDER items (ordered from supplier). "
        "Use those prices exactly as given — do NOT call any tool to look up prices or inventory. "
        "DISCOUNT RULE (apply this mechanically, no exceptions):\n"
        "  CUSTOMER PRICE (what the customer pays):\n"
        "    - qty >= 500: apply 10% bulk discount to customer price only.\n"
        "    - qty < 500: no discount.\n"
        "  SUPPLIER COST (what we pay the supplier = catalog unit price × qty):\n"
        "    - NEVER discounted, regardless of quantity.\n"
        "    - Always full catalog price. Supplier costs are internal — do not show them in the customer quote.\n"
        "Your quote shows ONLY CUSTOMER-FACING prices (with discount applied where qty>=500).\n"
        "   Never apply a discount based on combined totals across different items.\n"
        "Use search_quotes_tool to reference similar past quotes for consistency.\n"
        "For each line item: show base unit price, discount note (if applicable), and discounted subtotal.\n"
        "GRAND TOTAL IN QUOTE = SUM of all customer-facing subtotals only. "
        "Do NOT include supplier costs in this quote total.\n"
        "CRITICAL: Call final_answer as soon as you have the quote. Do NOT loop."
    ),
    max_steps=5,
)

order_agent = ToolCallingAgent(
    tools=[
        get_catalog_items_tool,
        normalize_item_name_tool,
        get_catalog_item_price_tool,
        cash_balance_tool,
        smart_order_tool,
    ],
    model=model,
    name="order_agent",
    description=(
        "You are the order_agent for The Beaver's Choice Paper Company. "
        "Your job is to record every transaction for the customer order.\n"
        "\n"
        "=== PROCESSING STEPS (follow in order) ===\n"
        "STEP 0 — Normalize: Call normalize_item_name_tool on EVERY item name. "
        "Use only the returned catalog name in all subsequent steps. "
        "Skip any item where normalize_item_name_tool returns 'Could not map'.\n"
        "STEP 1 — Price: Call get_catalog_item_price_tool to get the official unit price. "
        "Never use $0 or a guessed price.\n"
        "STEP 1.5 — Cash Check (REORDER only): Before placing any supplier REORDER, "
        "call cash_balance_tool with the order_date to check available funds. "
        "Calculate supplier cost = unit_price × quantity (supplier cost is NEVER discounted). "
        "RULE: Only proceed with the REORDER if available cash >= supplier cost. "
        "If available cash < supplier cost, DO NOT call smart_order_tool for that item. "
        "Instead, include a message in your final answer explaining that the item could not be "
        "reordered due to insufficient funds, and state the cash balance and required cost.\n"
        "STEP 2 — Order: Call smart_order_tool for EVERY item with these parameters:\n"
        "  - item_name          : normalized catalog name from STEP 0\n"
        "  - quantity           : units the customer wants (TOTAL quantity — see rule below)\n"
        "  - unit_price         : official price from STEP 1\n"
        "  - order_date         : REQUEST DATE from the task header (YYYY-MM-DD)\n"
        "  - customer_sale_price: same as unit_price (the tool will use it for the customer sale)\n"
        "  - delivery_date      : expected delivery date from the task (if provided), else leave empty\n"
        "  The tool AUTOMATICALLY decides SELL vs REORDER in code — you do NOT need to decide.\n"
        "  It also applies the 10% bulk discount automatically when quantity >= 500.\n"
        "  ██ ONE CALL PER ITEM — NO EXCEPTIONS ██\n"
        "  Call smart_order_tool EXACTLY ONCE per catalog item, using the FULL total quantity.\n"
        "  NEVER split one item into multiple calls (e.g., '500 reams' = one call with qty=500).\n"
        "  Splitting causes duplicate transactions and incorrect financial records.\n"
        "STEP 3 — Call final_answer with a clear summary of every item processed, "
        "the decision made (SELL or REORDER), and all transaction IDs."
    ),
    max_steps=15,
)

# ---------------------------------------------------------------------------
# 4. Orchestrator Agent
# (smolagents >= 1.0: pass CodeAgent objects directly to managed_agents —
#  no ManagedAgent wrapper needed)
# ---------------------------------------------------------------------------

orchestrator = ToolCallingAgent(
    tools=[get_catalog_items_tool, get_catalog_item_price_tool],
    model=model,
    managed_agents=[inventory_agent, quoting_agent, order_agent],
    name="orchestrator",
    instructions=(
        "You are The Beaver's Choice Paper Company AI assistant. "
        "You receive customer inquiries and coordinate the following specialist agents:\n"
        "  • inventory_agent — checks stock and delivery timelines\n"
        "  • quoting_agent   — produces competitive price quotes\n"
        "  • order_agent     — finalises orders and restocks inventory\n\n"
        "═══════════════════════════════════════════════════════\n"
        "STEP 0 — PARSE REQUEST (do this BEFORE calling any tool)\n"
        "═══════════════════════════════════════════════════════\n"
        "Read the customer request and list the DISTINCT PRODUCTS being ordered.\n"
        "CRITICAL PARSING RULES — violation of these is NOT allowed:\n"
        "  1. Adjectives such as 'colorful', 'colourful', 'assorted', 'recycled',\n"
        "     'large', 'heavy', 'standard', 'white', 'high-quality', 'biodegradable'\n"
        "     describe a product — they are NOT separate products themselves.\n"
        "  2. Treat the FULL phrase (adjective + noun) as ONE product name, then map\n"
        "     the noun to the catalog. Examples:\n"
        "       • '500 colorful poster paper'  → ONE item: 500 × Poster paper\n"
        "       • '300 colorful streamers'     → ONE item: 300 × Party streamers\n"
        "       • '1000 A4 glossy paper'       → ONE item: 1000 × Glossy paper\n"
        "       • '200 recycled cardstock'     → ONE item: 200 × Cardstock\n"
        "       • '500 biodegradable cups'     → ONE item: 500 × Paper cups\n"
        "  3. Do NOT split a single phrase into two products.\n"
        "     WRONG: '500 colorful poster paper' → Colored paper (500) + Poster paper (500)\n"
        "     RIGHT: '500 colorful poster paper' → Poster paper (500) only\n"
        "  4. Items not in the catalog (balloons, tickets, cardboard) → skip them\n"
        "     and inform the customer they are unavailable.\n"
        "Only after you have correctly identified each distinct product should you\n"
        "proceed to the numbered steps below.\n\n"
        "IMPORTANT: When calling managed agents, ALWAYS pass ONLY the 'task' argument as a plain string. "
        "Do NOT pass 'additional_args', 'request_date', or any other keyword arguments — "
        "embed all needed information (dates, quantities, prices) inside the task string itself.\n\n"
        "For every customer request you MUST follow these steps IN ORDER:\n"
        "1a. Call get_catalog_items_tool to retrieve all official catalog item names.\n"
        "1b. For EACH item the customer requests, call get_catalog_item_price_tool with the closest matching "
        "    EXACT catalog name. IMPORTANT: customers often use non-catalog names — you must map them:\n"
        "      - 'A4 glossy paper', 'A3 glossy paper', 'glossy A4 paper' → use catalog name 'Glossy paper'\n"
        "      - 'A4 matte paper', 'A3 matte paper', 'matte paper'     → use catalog name 'Matte paper'\n"
        "      - 'A4 paper', 'A3 paper', 'A4 white paper'              → use catalog name 'A4 paper'\n"
        "      - 'heavy cardstock', 'heavyweight cardstock'             → use catalog name 'Cardstock'\n"
        "      - 'colorful construction paper', 'colored construction'  → use catalog name 'Construction paper'\n"
        "      - 'standard printer paper', 'white printer paper'        → use catalog name 'Standard copy paper'\n"
        "      - 'washi tape', 'decorative washi tape'                  → use catalog name 'Decorative adhesive tape (washi tape)'\n"
        "      - 'colored paper', 'assorted colored paper'              → use catalog name 'Colored paper'\n"
        "      - 'recycled cardstock' or 'high-quality recycled'        → use catalog name 'Recycled paper' or 'Cardstock'\n"
        "      - poster boards / poster paper                           → use catalog name 'Poster paper' or 'Large poster paper (24x36 inches)'\n"
        "      - 'colorful poster paper', 'colourful poster paper'        → use catalog name 'Poster paper' (NOT Colored paper — 'colorful' is just an adjective here)\n"
        "      - 'colorful construction paper'                            → use catalog name 'Construction paper' (NOT Colored paper)\n"
        "      CRITICAL: when 'colorful'/'colourful' modifies another paper type, it is ONLY an adjective describing colour variety — do NOT treat it as a separate request for 'Colored paper'.\n"
        "    If after mapping the result from get_catalog_item_price_tool still says 'not found', "
        "    that item is NOT in our catalog — do NOT pass it to any sub-agent. "
        "    Tell the customer that specific item is unavailable from The Beaver's Choice.\n"
        "    ██ CRITICAL — PARTIAL ORDER RULE ██\n"
        "    Finding one unavailable item does NOT cancel or stop the rest of the order.\n"
        "    You MUST continue processing ALL remaining confirmed catalog items normally.\n"
        "    WRONG: Customer asks for Poster paper + Balloons → Balloons not found → you stop everything.\n"
        "    RIGHT: Customer asks for Poster paper + Balloons → Balloons not found → "
        "note 'Balloons unavailable', then continue steps 2-7 for Poster paper.\n"
        "    Only pass confirmed catalog items (with valid prices) to steps 2-7.\n"

        "2. Ask inventory_agent to check stock ONLY for confirmed catalog items. "
        "   Include the REQUEST DATE and all item names inside the task string. "
        "   ALWAYS pass the REQUEST DATE (not the customer's desired delivery date) as request_date, "
        "   so stock levels reflect what is available when the order is placed.\n"
        "3. Review inventory_agent's report to understand stock availability and delivery dates.\n"
        "   Non-catalog items: inform the customer they are unavailable.\n"
        "   NOTE: You do NOT need to classify items as SELL or REORDER — "
        "smart_order_tool inside order_agent makes that decision automatically in code.\n"

        "4. Ask quoting_agent to generate a CUSTOMER-FACING price quote using EXACTLY this task format:\n"
        "   'DISCOUNT RULE:\n"
        "    - The 10% bulk discount applies to CUSTOMER SALE PRICES (qty >= 500 per line item).\n"
        "    - This applies to ALL items the customer is buying — whether from stock (SELL) or ordered via supplier (REORDER).\n"
        "    - The supplier reorder cost is internal and NEVER discounted — but that is handled separately in order_agent.\n"
        "    - This quote shows only what the CUSTOMER pays.\n"
        "    Generate a customer price quote for:\n"
        "    SELL (fulfilled from stock):\n"
        "    - <item_name>: qty=<N> units at $<price> each\n"
        "    (or: None)\n"
        "    REORDER (will be delivered from supplier — customer charged on delivery date):\n"
        "    - <item_name>: qty=<N> units at $<price> each\n"
        "    (or: None)\n"
        "    For each line: show base price, discount (if qty>=500), discounted subtotal.\n"
        "    Show grand total = sum of all customer-facing subtotals.'\n"
        "   Do NOT include stock counts, delivery dates, supplier costs, or reorder flags in the quoting task.\n"
        "5. Call inventory_agent to get the supplier delivery date for ALL confirmed catalog items "
        "(not just REORDER — smart_order_tool needs delivery_date for every item in case it decides to reorder). "
        "Pass the request date and the full item list.\n"
        "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
        "6. ██ MANDATORY ██ Call order_agent EXACTLY ONCE — NO EXCEPTIONS.\n"
        "   Skipping this step is NOT allowed, even if:\n"
        "     • ALL items appear to be in stock\n"
        "     • The customer only asked for a quote\n"
        "     • You already have pricing from quoting_agent\n"
        "   A quote without a recorded transaction has ZERO business value.\n"
        "   You MUST call order_agent after step 5, always.\n"
        "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
        "   Build the task string in this format:\n"
        "   REQUEST DATE: <YYYY-MM-DD>\n"
        "   ITEMS:\n"
        "   - <catalog_item_name>, qty=<N>, unit_price=$<P>, customer_sale_price=$<P>, "
        "delivery_date=<YYYY-MM-DD from step 5>\n"
        "   (repeat for every confirmed catalog item)\n"
        "   RULES:\n"
        "   - Use ONLY exact official catalog names from get_catalog_item_price_tool\n"
        "   - customer_sale_price = same as unit_price (bulk discount applied by tool)\n"
        "   - delivery_date = date from inventory_agent step 5 (NEVER guess a date)\n"
        "   - smart_order_tool will decide SELL vs REORDER automatically in Python code\n"
        "   Never call order_agent more than once.\n"

        "CRITICAL RULE: Never tell the customer an order is placed unless order_agent has explicitly confirmed it. "
        "You CANNOT fulfill orders or check inventory yourself — always delegate.\n"
        "7. Return a clear, friendly FINAL RESPONSE using this exact structure:\n"
        "\n"
        "   ## Order Summary\n"
        "   List every catalog item with:\n"
        "   - Item name, quantity\n"
        "   - Status: SELL (from stock) or REORDER (from supplier)\n"
        "   - Transaction ID(s) from order_agent\n"
        "   - Delivery date (for REORDER items)\n"
        "\n"
        "   ## Financial Breakdown\n"
        "   Show ALL inflows (+) and outflows (-) using this EXACT format:\n"
        "\n"
        "   INFLOWS (what the customer pays to us):\n"
        "   + <Item>: <qty> units × $<discounted_unit_price> = +$<subtotal>  [10% discount applied / no discount]\n"
        "   (one line per item sold or reordered for a customer)\n"
        "\n"
        "   OUTFLOWS (what we pay to the supplier):\n"
        "   - <Item>: <qty> units × $<catalog_unit_price> = -$<subtotal>  [supplier cost, no discount]\n"
        "   (one line per REORDER item only — SELL items have no outflow since stock is already owned)\n"
        "\n"
        "   NET BALANCE = Total Inflows - Total Outflows\n"
        "   (positive = net revenue this order; negative = cost exceeds revenue this order)\n"
        "\n"
        "   Rules for the financial breakdown:\n"
        "   • SELL items: inflow only (+ customer revenue). No outflow — stock was already purchased.\n"
        "   • REORDER items: BOTH an inflow (+ customer pays discounted price) AND an outflow (- supplier cost at full catalog price).\n"
        "   • Supplier outflow is ALWAYS full catalog price — NEVER discounted.\n"
        "   • Customer inflow is discounted price if qty >= 500, else full price.\n"
        "   • NEVER add supplier costs to the customer subtotals. They are separate columns.\n"
        "   • Items not in the catalog must be EXPLICITLY listed as unavailable — do not silently drop them."
    ),
    max_steps=14,
)


def call_orchestrator(request_text: str) -> str:
    """
    Entry point: send a customer request to the orchestrator and return its response.

    Args:
        request_text: Full customer request string (must include the date).

    Returns:
        The orchestrator's final text response.
    """
    try:
        result = orchestrator.run(request_text)
        return str(result)
    except Exception as exc:
        return f"[Agent error] {exc}"


# ---------------------------------------------------------------------------
# 6. Test runner
# ---------------------------------------------------------------------------

def run_test_scenarios():

    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv(os.path.join(BASE_DIR, "quote_requests_sample.csv"))
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    # Orchestrator is already initialised at module level (see above)

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Build the full request string (includes date so agents can act on it)
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        # ── Call the multi-agent orchestrator ──────────────────────────────
        response = call_orchestrator(request_with_date)
        # ───────────────────────────────────────────────────────────────────

        # Update state from DB after the orchestrator may have committed transactions
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
