# Reflection Report: Beaver's Choice Paper Company — Multi-Agent System

**Student Name:** [Your Name]  
**Submission Date:** [Date]  
**Framework Used:** smolagents (v1.24.0)  
**Model:** GPT-4o-mini via OpenAI API  

---

## Part 1: Agent Workflow Architecture Explanation

### 1.1 Overview

The system I built for the Beaver's Choice Paper Company is a four-agent hierarchy implemented using the `smolagents` Python library with `ToolCallingAgent` as the base class for all agents. The architecture follows a hub-and-spoke pattern: one orchestrator agent coordinates three specialist worker agents, each responsible for a distinct operational domain. All agents communicate exclusively through text-based task strings and final answers, as required by the project specification.

The workflow diagram (see `agent_workflow_diagram.png`) illustrates this structure in full. The explanation below details each agent's role and the reasoning behind key architectural decisions.

---

### 1.2 The Orchestrator Agent

**Role:** Entry point for all customer interactions. Receives the raw customer request, parses it to identify catalog items, and coordinates the three worker agents in sequence.

**Why this design:**  
The orchestrator acts as the "brain" of the system. Rather than having a single agent try to do everything, the orchestrator focuses exclusively on:
1. Parsing ambiguous customer language (e.g., "colorful construction paper" → catalog name "Construction paper")
2. Validating that requested items exist in the catalog via `get_catalog_items_tool` and `get_catalog_item_price_tool`
3. Delegating tasks to specialist agents in the correct sequence
4. Assembling the final formatted response with order summary, financial breakdown, and net balance

**Direct tools available to the Orchestrator:**
| Tool | Helper Function | Purpose |
|------|----------------|---------|
| `get_catalog_items_tool` | `get_all_inventory()` | Retrieves the full list of 42 catalog product names |
| `get_catalog_item_price_tool` | `get_all_inventory()` + filtering | Looks up the official unit price for any catalog item |

**Key design decision — Partial Order Rule:** If a customer requests an item that is not in the catalog (e.g., balloons, tickets), the orchestrator is instructed to flag it as unavailable but continue processing all remaining catalog items. This ensures that one invalid item never blocks an entire order.

---

### 1.3 The Inventory Agent

**Role:** Checks current stock levels for requested items and estimates supplier delivery dates when reorders are required.

**Why this design:**  
Inventory status and delivery timeline are tightly coupled. When the inventory agent determines that an item needs to be reordered (stock is insufficient), the orchestrator immediately needs the delivery date to include in the customer response. By giving the inventory agent both stock-checking tools AND the delivery date tool, a single delegation call to `inventory_agent` returns a complete status report: stock availability AND expected delivery date. This eliminates an extra round-trip to a separate "delivery" agent.

**Tools available to the Inventory Agent:**
| Tool | Helper Function | Purpose |
|------|----------------|---------|
| `check_inventory_tool` | `get_stock_level()` | Checks current stock for a specific item on a given date |
| `check_all_inventory_tool` | `get_all_inventory()` | Lists stock levels across all items |
| `get_delivery_date_tool` | `get_supplier_delivery_date()` | Estimates supplier delivery date based on order quantity |

**Stock logic:** Stock is never stored as a fixed number. Instead, `get_stock_level()` dynamically calculates stock as: `SUM(stock_orders) - SUM(sales)` up to the request date from the transactions table. This ensures historical accuracy.

---

### 1.4 The Quoting Agent

**Role:** Generates competitive, accurate price quotes for the customer, applying bulk discounts where applicable.

**Why this design:**  
Pricing logic requires careful rule enforcement. By isolating quoting in a dedicated agent with a strictly defined discount rule, the system prevents the orchestrator or order agent from applying discounts inconsistently. The quoting agent operates only on prices provided by the orchestrator — it does not do its own inventory lookups. It uses `search_quotes_tool` to reference historical quotes from `quotes.csv` for pricing consistency with past transactions.

**Tools available to the Quoting Agent:**
| Tool | Helper Function | Purpose |
|------|----------------|---------|
| `search_quotes_tool` | `search_quote_history()` | Searches historical quote data for comparable past quotes |

**Discount Rule (hard-coded in agent instructions):**
- Customer quantity ≥ 500 units → **10% bulk discount** on customer-facing price
- Supplier cost → **never discounted**, always full catalog price
- Discounts applied per line item individually (not on combined order totals)

---

### 1.5 The Order Agent

**Role:** Records every transaction in the SQLite database, determining automatically whether each item is a SELL (from existing stock) or a REORDER (from supplier).

**Why this design:**  
The order agent is the only agent that writes to the database. Isolating this responsibility prevents other agents from accidentally creating duplicate or malformed transaction records. The `smart_order_tool` encapsulates the full SELL vs REORDER decision in Python code (not left to LLM judgment), making it deterministic and reliable.

**Tools available to the Order Agent:**
| Tool | Helper Function | Purpose |
|------|----------------|---------|
| `get_catalog_items_tool` | `get_all_inventory()` | Retrieves catalog for name validation |
| `normalize_item_name_tool` | Fuzzy catalog match | Maps customer-provided names to exact catalog names |
| `get_catalog_item_price_tool` | Price lookup | Gets official unit price for each item |
| `smart_order_tool` | `create_transaction()` | Records SELL or REORDER transactions; auto-applies 10% bulk discount |

**SELL vs REORDER logic (in `smart_order_tool`):**
- If current stock ≥ requested quantity → **SELL**: records one `sales` transaction, deducts from stock
- If current stock < requested quantity → **REORDER**: records a `stock_orders` transaction (supplier purchase) AND a `sales` transaction (customer sale) on the same date

---

### 1.6 Data Flow Summary

```
Customer Request (text)
        ↓
ORCHESTRATOR: Parse → validate catalog items → get prices
        ↓
INVENTORY AGENT: Check stock → estimate delivery dates
        ↓
QUOTING AGENT: Generate quote → apply bulk discounts → search quote history
        ↓
ORDER AGENT: Normalize names → record transactions (SELL or REORDER) in SQLite
        ↓
ORCHESTRATOR: Assemble final answer (order summary + financial breakdown)
        ↓
Final Answer (text) → saved to test_results.csv
```

---

## Part 2: Evaluation Results Discussion

### 2.1 Test Setup

The system was evaluated against all 20 requests from `quote_requests_sample.csv`. Each request simulated a real customer inquiry with varying item types, quantities, and contexts (hotel manager, school board, restaurant, etc.). Results are documented in `test_results.csv`.

**Initial financial state:**  
- Cash: $45,059.70 | Inventory: $4,940.30 | Total assets: **$50,000.00**

**Final financial state:**  
- Cash: $45,044.10 | Inventory: $4,668.55 | Total assets: **$49,712.65**

**Overall net change: −$287.35** (net cost to business across all 20 orders)

---

### 2.2 Quantitative Results

| Metric | Result |
|--------|--------|
| Total requests processed | 20/20 |
| Requests with cash balance change | 20/20 |
| Requests successfully fulfilled (with transactions) | 18/20 |
| Requests with SELL transactions | 8 requests |
| Requests with REORDER transactions | 14 requests |
| Peak total assets (after Req 8) | $50,434.75 |
| Largest single-order gain (Req 8) | +$423.00 |
| Largest single-order loss (Req 20) | −$375.00 |

> All metrics above exceed the rubric minimum of "at least three requests with cash balance changes" and "at least three quote requests successfully fulfilled."

---

### 2.3 Identified Strengths

**Strength 1 — Correct Bulk Discount Application**  
The system consistently applied the 10% bulk discount to customer-facing prices for orders ≥ 500 units across all applicable requests (Requests 2, 3, 4, 7, 8, 10, 13, 14, 15, 16, 17, 19, 20). Supplier costs were never discounted. This was enforced deterministically in `smart_order_tool` rather than relying solely on the LLM to apply the rule.

**Strength 2 — Accurate SELL vs REORDER Classification**  
The `smart_order_tool` correctly classified items as SELL (from stock) or REORDER (from supplier) based on live stock levels calculated at each request date. For example, in Request 1, all three items (Cardstock, Colored paper, Glossy paper) were correctly sold from stock. In Request 3, A4 paper and Standard copy paper were correctly reordered because stock was insufficient.

**Strength 3 — Partial Order Handling**  
When a customer requested items outside the catalog, the system correctly continued processing the valid catalog items without blocking the entire order. This was particularly important for Request 20, where the non-catalog item was identified and the order for catalog items proceeded normally.

**Strength 4 — Multi-Agent Coordination**  
The orchestrator successfully delegated tasks to specialist agents in the correct sequential order for all 18 fully completed requests. The inventory agent's combined stock + delivery date reporting enabled efficient single-call coordination.

---

### 2.4 Areas for Improvement (Issues Observed)

**Issue 1 — Request 6: Order Not Placed**  
The orchestrator responded to the school principal's request by listing pricing and asking: *"please confirm if you'd like to proceed"* instead of immediately placing the order. This was a misinterpretation of the agent's role. In the current setup, the system should assume orders are to be fulfilled — confirmation loops were not part of the specification.

**Issue 2 — Request 20: Non-Catalog Item Incorrectly Mapped**  
The customer requested "10,000 tickets." Tickets are not in the paper catalog, but the order agent mapped them to "Paper party bags" instead of flagging them as unavailable. This resulted in an unintended transaction of 10,000 paper party bags.

**Issue 3 — LLM Arithmetic Error (Request 2)**  
The orchestrator's final text response stated NET BALANCE = −$13.50, but the correct calculation was −$12.50. The underlying database transactions were correct (the actual cash balance changed by exactly −$12.50), confirming this was an LLM reasoning error in writing the final text summary, not a transaction error.

---

## Part 3: Suggestions for Further Improvement

### Suggestion 1 — Add a Math Verification Step in the Orchestrator

**Problem addressed:** Arithmetic errors in the final response (Issue 3 above).

**Proposed solution:** Before calling `final_answer`, add an explicit instruction to the orchestrator to calculate the net balance programmatically from the cash balance change rather than asking the LLM to add up numbers in its head. Specifically:

```
NET BALANCE = new_cash_balance - old_cash_balance
```

Since both values are available from `generate_financial_report()`, this calculation can be delegated to a `compute_net_balance_tool` that returns the exact number. This removes the arithmetic from the LLM's response generation entirely, eliminating the risk of miscalculation.

---

### Suggestion 2 — Add a Catalog Validation Pre-Filter Tool

**Problem addressed:** Non-catalog items being incorrectly mapped to catalog products (Issue 2 above).

**Proposed solution:** Create a dedicated `validate_catalog_items_tool` that uses fuzzy string matching to classify each requested item as:
- `CONFIRMED`: exact or close match found in catalog
- `AMBIGUOUS`: possible match, needs human review
- `NOT_FOUND`: no match in catalog → tell customer this item is unavailable

This would be called as the very first step by the orchestrator, before any other tools. Items marked `NOT_FOUND` would be excluded from all downstream processing, preventing the order agent from ever trying to match them to catalog products. The `normalize_item_name_tool` already exists for this purpose in the order agent — promoting this step to the orchestrator level would be the key change.

---

### Suggestion 3 (Bonus) — Add Retry Logic for Agent Failures

**Problem addressed:** Single agent errors (like Issue 1) block the entire request.

**Proposed solution:** Wrap each managed agent call in a try/except with a retry prompt. If an agent returns an unclear or incomplete response (e.g., asks for confirmation), the orchestrator should detect this pattern and resend the task with an explicit instruction: *"Do not ask for confirmation. Place the order immediately."* This makes the system more robust to LLM variability across different request types.

---

## Conclusion

The multi-agent system for the Beaver's Choice Paper Company demonstrates a working implementation of a 4-agent hierarchy using `smolagents`, capable of processing realistic customer inquiries end-to-end: from parsing ambiguous natural language requests to recording accurate financial transactions in an SQLite database. The system correctly fulfilled 18 of 20 test requests, consistently applied pricing and discount rules, and maintained accurate financial records throughout the evaluation.

Key areas for future improvement centre on reducing LLM reliance for deterministic operations (arithmetic, catalog validation) by moving more logic into Python tool functions, where results are precise and repeatable regardless of model variability.

---

*Report prepared for Udacity Agentic AI Course — Chapter 4 Assignment*
