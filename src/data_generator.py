"""
Data Generator - builds the sample customer histories the system works over.

The output is deterministic: a fixed seed means the committed
data/sample_customers.json can always be regenerated exactly, so a reviewer can
check the data against the code that produced it.

Each customer is a persona - a salary cadence, a set of recurring bills, some
discretionary spending - and a handful of them carry a deliberately planted
anomaly so that every risk rule has something real to find.
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

from src.utils import make_transaction_id

SEED = 20240904
BUSINESS_HOURS = (8, 21)


def _money(value: float) -> float:
    return round(value, 2)


def _txn(rng, day: date, payee: str, amount: float, channel: str,
         kind: str, description: str, hour_range=BUSINESS_HOURS, timestamp=None):
    if timestamp is None:
        hour = rng.randint(*hour_range)
        timestamp = f"{hour:02d}:{rng.randint(0, 59):02d}"
    return {
        "date": day.isoformat(),
        "payee": payee,
        "amount": _money(amount),
        "channel": channel,
        "type": kind,
        "description": description,
        "timestamp": timestamp,
    }


def _month_starts(start: date, months: int):
    day = start
    for _ in range(months):
        yield day
        year, month = day.year, day.month + 1
        if month > 12:
            year, month = year + 1, 1
        day = date(year, month, min(day.day, 28))


def build_history(rng, start: date, months: int, salary, recurring, discretionary):
    """Assemble one customer's routine activity across the observation window."""
    txns = []

    for month_start in _month_starts(start, months):
        if salary:
            payee, amount, channel, paydays = salary
            for offset in paydays:
                day = month_start + timedelta(days=offset)
                txns.append(_txn(rng, day, payee, amount, channel, "credit", "Salary Deposit"))

        for payee, amount, offset, channel, description in recurring:
            day = month_start + timedelta(days=offset)
            jitter = rng.uniform(-0.03, 0.03)
            txns.append(_txn(rng, day, payee, amount * (1 + jitter), channel, "debit", description))

        for payee, low, high, per_month, channel, description in discretionary:
            for _ in range(per_month):
                day = month_start + timedelta(days=rng.randint(0, 27))
                txns.append(_txn(rng, day, payee, rng.uniform(low, high), channel, "debit", description))

    return txns


def finalise(customer_id: str, account_type: str, txns):
    """
    Sort the history and stamp each row with its identifier.

    The ids are written into the data itself rather than assigned at load time,
    so that when a finding cites a transaction the reference can be found in the
    committed source history by searching for it.
    """
    txns = sorted(txns, key=lambda t: (t["date"], t["timestamp"]))
    for index, txn in enumerate(txns):
        txn["transaction_id"] = make_transaction_id(
            customer_id, index, txn["date"], txn["payee"], txn["amount"]
        )
    return {"customer_id": customer_id, "account_type": account_type, "transactions": txns}


def generate_sample_customers():
    rng = random.Random(SEED)
    start = date(2024, 1, 2)
    customers = []

    # ---- CUST_001: salaried employee, entirely routine -------------------
    txns = build_history(
        rng, start, 5,
        salary=("Northwind Systems", 3500.00, "ACH", [0, 14]),
        recurring=[
            ("Riverside Electric", 125.00, 4, "ACH", "Utility Bill"),
            ("Metro Water", 48.00, 9, "ACH", "Utility Bill"),
            ("Netflix", 15.99, 11, "Online", "Subscription"),
            ("Union Mortgage", 1450.00, 1, "ACH", "Mortgage Payment"),
        ],
        discretionary=[
            ("Whole Foods", 62.0, 105.0, 3, "Debit Card", "Grocery Shopping"),
            ("Shell Station", 38.0, 62.0, 2, "Debit Card", "Fuel"),
            ("Corner Pharmacy", 18.0, 45.0, 1, "Debit Card", "Prescription"),
        ],
    )
    customers.append(finalise("CUST_001", "Checking", txns))

    # ---- CUST_002: small business, large but consistent amounts ----------
    txns = build_history(
        rng, start, 5,
        salary=None,
        recurring=[
            ("Sterling Supply Co", 1200.00, 2, "ACH", "Inventory Purchase"),
            ("Harbour Business Park", 2200.00, 1, "ACH", "Commercial Rent"),
            ("Utilities Business", 450.00, 9, "ACH", "Utility Bill"),
        ],
        discretionary=[
            ("Office Depot", 180.0, 420.0, 2, "Debit Card", "Office Supplies"),
            ("Sterling Supply Co", 900.0, 1500.0, 1, "ACH", "Inventory Purchase"),
        ],
    )
    for month_start in _month_starts(start, 5):
        txns.append(_txn(rng, month_start + timedelta(days=3), "Aldridge Client Corp",
                         rng.uniform(4800, 5400), "ACH", "credit", "Invoice Payment"))
        txns.append(_txn(rng, month_start + timedelta(days=17), "Bellamy Client Inc",
                         rng.uniform(6800, 7900), "ACH", "credit", "Invoice Payment"))
    customers.append(finalise("CUST_002", "Checking", txns))

    # ---- CUST_003: routine, then a burst of wires to a brand-new payee ---
    txns = build_history(
        rng, start, 5,
        salary=("Acme Manufacturing", 4000.00, "ACH", [0, 14]),
        recurring=[
            ("City Power", 140.00, 8, "ACH", "Electric Bill"),
            ("Greenway Apartments", 1300.00, 1, "ACH", "Rent Payment"),
            ("Netflix", 16.99, 20, "Online", "Subscription"),
        ],
        discretionary=[
            ("Safeway", 58.0, 96.0, 3, "Debit Card", "Groceries"),
            ("Chevron", 40.0, 58.0, 2, "Debit Card", "Fuel"),
        ],
    )
    burst_day = date(2024, 5, 12)
    for offset, amount in ((0, 2500.00), (1, 1500.00), (2, 1000.00)):
        txns.append(_txn(rng, burst_day + timedelta(days=offset), "CryptoExchange XYZ",
                         amount, "Wire", "debit", "Outgoing Wire Transfer"))
    customers.append(finalise("CUST_003", "Checking", txns))

    # ---- CUST_004: routine, then one very large wire out -----------------
    txns = build_history(
        rng, start, 5,
        salary=("Vertex Technologies", 5000.00, "ACH", [3, 18]),
        recurring=[
            ("Lakeside Rentals", 1650.00, 1, "ACH", "Rent Payment"),
            ("City Power", 118.00, 7, "ACH", "Electric Bill"),
        ],
        discretionary=[
            ("Walmart", 70.0, 130.0, 3, "Debit Card", "Shopping"),
            ("Shell Station", 42.0, 66.0, 2, "Debit Card", "Fuel"),
        ],
    )
    txns.append(_txn(rng, date(2024, 5, 16), "Meridian Overseas Bank", 12000.00,
                     "Wire", "debit", "International Wire Transfer"))
    customers.append(finalise("CUST_004", "Checking", txns))

    # ---- CUST_005: EDGE CASE - large transfers that are the norm ---------
    # A monthly investment sweep, every month, to the same brokerage. Large in
    # absolute terms but established: the system should leave this alone.
    txns = build_history(
        rng, start, 5,
        salary=("Halden Finance", 6000.00, "ACH", [3, 18]),
        recurring=[
            ("Ashford Property Mgmt", 2000.00, 12, "ACH", "Monthly Rent Payment"),
            ("Fairmont Brokerage", 5000.00, 20, "Wire", "Scheduled Investment Transfer"),
        ],
        discretionary=[
            ("Trader Joes", 90.0, 150.0, 2, "Debit Card", "Groceries"),
        ],
    )
    customers.append(finalise("CUST_005", "Checking", txns))

    # ---- CUST_006: very clean, low-variance pattern ----------------------
    txns = build_history(
        rng, start, 5,
        salary=("Cordell Corp", 3200.00, "ACH", [0, 14]),
        recurring=[
            ("Anytime Fitness", 50.00, 10, "ACH", "Membership Fee"),
            ("Brookline Apartments", 1150.00, 1, "ACH", "Rent Payment"),
        ],
        discretionary=[
            ("Daily Grind Coffee", 4.5, 7.0, 4, "Debit Card", "Coffee"),
            ("Kroger", 55.0, 78.0, 2, "Debit Card", "Groceries"),
        ],
    )
    customers.append(finalise("CUST_006", "Checking", txns))

    # ---- CUST_007: pattern break - money starts leaving by new routes ----
    txns = build_history(
        rng, start, 5,
        salary=("Bancroft Group", 4500.00, "ACH", [0, 14]),
        recurring=[
            ("Parkview Leasing", 1400.00, 1, "ACH", "Rent Payment"),
            ("City Power", 150.00, 7, "ACH", "Electric Bill"),
        ],
        discretionary=[
            ("Target", 85.0, 135.0, 3, "Debit Card", "Shopping"),
        ],
    )
    # The closing weeks turn over almost entirely: unfamiliar names, unfamiliar
    # routes. No single payee repeats, so this is a change of pattern rather
    # than a burst - it is the shape of the activity that is wrong, not any one
    # payment in it.
    for offset, (payee, amount, channel, desc) in enumerate((
        ("Prepaid Card Reload", 260.00, "ATM", "Prepaid Card Load"),
        ("Instant Money Remit", 310.00, "Wire", "Remittance"),
        ("Voucher Direct", 180.00, "Online", "Gift Voucher Purchase"),
        ("TopUp Wallet Services", 240.00, "Online", "Wallet Top-Up"),
        ("Global Send Agency", 295.00, "Wire", "Remittance"),
        ("CashPoint Exchange", 220.00, "ATM", "Currency Exchange"),
        ("Rapid Transfer Ltd", 275.00, "Wire", "Outgoing Transfer"),
        ("Digital Voucher Hub", 160.00, "Online", "Gift Voucher Purchase"),
        ("Swift Remit Partners", 330.00, "Wire", "Remittance"),
    )):
        txns.append(_txn(rng, date(2024, 6, 1) + timedelta(days=offset),
                         payee, amount, channel, "debit", desc))
    customers.append(finalise("CUST_007", "Checking", txns))

    # ---- CUST_008: odd-hours activity ------------------------------------
    txns = build_history(
        rng, start, 5,
        salary=("Ridgeline Logistics", 3800.00, "ACH", [1, 15]),
        recurring=[
            ("Fielding Estates", 1250.00, 1, "ACH", "Rent Payment"),
            ("City Power", 115.00, 7, "ACH", "Electric Bill"),
        ],
        discretionary=[
            ("Publix", 85.0, 120.0, 3, "Debit Card", "Groceries"),
            ("Corner Pharmacy", 25.0, 55.0, 1, "Debit Card", "Prescription"),
        ],
    )
    for offset, (amount, ts) in enumerate((
        (420.00, "03:14"), (380.00, "02:47"), (455.00, "04:05"),
    )):
        txns.append(_txn(rng, date(2024, 5, 9) + timedelta(days=offset),
                         "QuickPay Transfer", amount, "Online", "debit",
                         "Peer-to-Peer Transfer", timestamp=ts))
    customers.append(finalise("CUST_008", "Checking", txns))

    # ---- CUST_009: several signals at once -> escalation ------------------
    txns = build_history(
        rng, start, 5,
        salary=("Kestrel Software", 5500.00, "ACH", [0, 14]),
        recurring=[
            ("Highgate Residences", 1800.00, 1, "ACH", "Rent Payment"),
            ("Fiber Internet Co", 60.00, 7, "ACH", "Monthly Fee"),
        ],
        discretionary=[
            ("Daily Grind Coffee", 4.5, 7.5, 3, "Debit Card", "Coffee"),
            ("Uber Eats", 22.0, 48.0, 2, "Online", "Food Delivery"),
        ],
    )
    # A burst to one new payee, at night, one leg of it far above normal size.
    for offset, (amount, ts) in enumerate((
        (3500.00, "03:22"), (2800.00, "02:58"), (2200.00, "03:41"),
    )):
        txns.append(_txn(rng, date(2024, 5, 14) + timedelta(days=offset),
                         "Vantage Holdings Ltd", amount, "Wire", "debit",
                         "Outgoing Wire Transfer", timestamp=ts))
    customers.append(finalise("CUST_009", "Checking", txns))

    # ---- CUST_010: too little history to judge ---------------------------
    txns = [
        _txn(rng, date(2024, 4, 3), "Savings Transfer In", 2000.00, "ACH", "credit", "Deposit"),
        _txn(rng, date(2024, 4, 11), "Savings Transfer Out", 500.00, "ACH", "debit", "Withdrawal"),
        _txn(rng, date(2024, 4, 19), "Savings Transfer In", 2000.00, "ACH", "credit", "Deposit"),
        _txn(rng, date(2024, 4, 26), "Savings Transfer Out", 500.00, "ACH", "debit", "Withdrawal"),
        _txn(rng, date(2024, 5, 2), "Savings Transfer In", 2000.00, "ACH", "credit", "Deposit"),
    ]
    customers.append(finalise("CUST_010", "Savings", txns))

    # ---- CUST_011 .. CUST_018: routine personas --------------------------
    personas = [
        ("CUST_011", "Checking", ("Lumen Startup", 8000.00, "ACH", [0, 14]),
         [("Marlowe Towers", 2600.00, 1, "ACH", "Rent Payment"),
          ("Elite Athletic Club", 150.00, 9, "ACH", "Membership Fee")],
         [("Blue Bottle Coffee", 9.0, 14.0, 3, "Debit Card", "Coffee"),
          ("Nobu Restaurant", 180.0, 290.0, 2, "Debit Card", "Dining")]),

        ("CUST_012", "Checking", ("State Benefits Office", 2500.00, "ACH", [1, 15]),
         [("Housing Authority", 1200.00, 7, "ACH", "Rent"),
          ("City Power", 78.00, 12, "ACH", "Electric Bill")],
         [("Aldi", 38.0, 62.0, 3, "Debit Card", "Groceries"),
          ("Transit Authority", 20.0, 30.0, 2, "Debit Card", "Transit Pass")]),

        ("CUST_013", "Checking", None,
         [("Coworking Loft", 800.00, 2, "ACH", "Workspace Rental"),
          ("Adobe Creative Cloud", 54.00, 8, "Online", "Subscription")],
         [("Kroger", 60.0, 95.0, 2, "Debit Card", "Groceries"),
          ("Amtrak", 90.0, 180.0, 1, "Online", "Travel")]),

        ("CUST_014", "Checking", ("Campus Employment Office", 1400.00, "ACH", [2, 16]),
         [("University Housing", 620.00, 1, "ACH", "Dorm Fees"),
          ("Student Health Plan", 45.00, 10, "ACH", "Insurance")],
         [("Campus Cafeteria", 12.0, 26.0, 4, "Debit Card", "Meals"),
          ("Steam Games", 20.0, 60.0, 1, "Online", "Entertainment")]),

        ("CUST_015", "Checking", None,
         [("Fairmont Brokerage", 8000.00, 5, "Wire", "Bond Investment"),
          ("Wealth Advisory Fees", 400.00, 12, "ACH", "Advisory Fee")],
         [("Whole Foods", 120.0, 190.0, 2, "Debit Card", "Groceries")]),

        ("CUST_016", "Checking", ("Mercy Health Group", 4200.00, "ACH", [0, 14]),
         [("Sunnyvale Rentals", 1500.00, 1, "ACH", "Rent Payment"),
          ("Corner Pharmacy", 65.00, 9, "Debit Card", "Prescription")],
         [("Safeway", 70.0, 110.0, 3, "Debit Card", "Groceries"),
          ("Shell Station", 40.0, 60.0, 2, "Debit Card", "Fuel")]),

        ("CUST_017", "Checking", ("Corporate HQ Payroll", 6500.00, "ACH", [1, 15]),
         [("Ashcroft Mortgage", 2100.00, 1, "ACH", "Mortgage Payment"),
          ("Statewide Auto Insurance", 120.00, 11, "ACH", "Monthly Premium")],
         [("Shell Station", 55.0, 72.0, 3, "Debit Card", "Fuel"),
          ("Home Depot", 60.0, 210.0, 1, "Debit Card", "Home Improvement")]),

        ("CUST_018", "Checking", ("Fairview Retail", 3500.00, "ACH", [0, 14]),
         [("Cedar Court Apartments", 1100.00, 1, "ACH", "Rent Payment"),
          ("City Power", 100.00, 8, "ACH", "Electric Bill")],
         [("Kroger", 65.0, 92.0, 3, "Debit Card", "Groceries"),
          ("Old Navy", 40.0, 110.0, 1, "Debit Card", "Clothing")]),
    ]
    for cid, account_type, salary, recurring, discretionary in personas:
        txns = build_history(rng, start, 5, salary, recurring, discretionary)
        if salary is None:  # freelance / investment income arrives irregularly
            for month_start in _month_starts(start, 5):
                txns.append(_txn(rng, month_start + timedelta(days=rng.randint(2, 20)),
                                 "Client Retainer", rng.uniform(3000, 5200),
                                 "ACH", "credit", "Project Payment"))
        customers.append(finalise(cid, account_type, txns))

    # ---- CUST_019: barely any history at all -----------------------------
    txns = [
        _txn(rng, date(2024, 4, 5), "Regular Deposit", 1000.00, "ACH", "credit", "Savings Deposit"),
        _txn(rng, date(2024, 4, 20), "Regular Deposit", 1000.00, "ACH", "credit", "Savings Deposit"),
        _txn(rng, date(2024, 5, 6), "Regular Deposit", 1000.00, "ACH", "credit", "Savings Deposit"),
    ]
    customers.append(finalise("CUST_019", "Savings", txns))

    # ---- CUST_020: a young account - enough activity, not enough time ----
    # Deliberately free of any large recurring obligation. A first rent payment
    # in a three-week history is indistinguishable from an unusual transfer, and
    # this customer exists to show the EMERGING path returning clean rather than
    # to test how the system handles that ambiguity.
    txns = build_history(
        rng, date(2024, 5, 1), 1,
        salary=("Tessellate Labs", 3400.00, "ACH", [2, 16]),
        recurring=[
            ("Cloud Services", 25.00, 6, "Online", "Subscription"),
            ("Metro Transit", 60.00, 9, "ACH", "Monthly Pass"),
        ],
        discretionary=[
            ("Best Buy", 120.0, 190.0, 3, "Debit Card", "Electronics"),
            ("Uber Eats", 25.0, 55.0, 5, "Online", "Food Delivery"),
            ("Kroger", 45.0, 80.0, 3, "Debit Card", "Groceries"),
        ],
    )
    customers.append(finalise("CUST_020", "Checking", txns))

    return customers


if __name__ == "__main__":
    data = generate_sample_customers()
    out = Path(__file__).resolve().parent.parent / "data" / "sample_customers.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    total = sum(len(c["transactions"]) for c in data)
    print(f"Wrote {len(data)} customers, {total} transactions -> {out}")
