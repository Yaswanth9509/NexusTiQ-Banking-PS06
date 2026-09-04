"""Shared fixtures."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analyzer import TransactionAnalyzer  # noqa: E402
from src.models import CustomerHistory, Transaction  # noqa: E402
from src.rules import (  # noqa: E402
    BurstToNewPayeeRule,
    OddHoursActivityRule,
    PatternBreakRule,
    UnusuallyLargeTransferRule,
)


@pytest.fixture(scope="session")
def sample_records():
    """The committed sample histories, as raw dictionaries."""
    path = ROOT / "data" / "sample_customers.json"
    return {c["customer_id"]: c for c in json.loads(path.read_text(encoding="utf-8"))}


@pytest.fixture(scope="session")
def histories(sample_records):
    return {cid: CustomerHistory(**record) for cid, record in sample_records.items()}


@pytest.fixture
def analyzer():
    return TransactionAnalyzer(
        risk_rules=[
            UnusuallyLargeTransferRule(multiplier=2.5),
            BurstToNewPayeeRule(window_days=7, min_transactions=2),
            OddHoursActivityRule(odd_hour_start=0, odd_hour_end=6, threshold=2),
            PatternBreakRule(),
        ]
    )


def run(coro):
    """Drive a coroutine to completion.

    Used instead of pytest-asyncio: the analysis entry point is async but does
    no I/O, so a plugin dependency would buy nothing on a clean machine.
    """
    return asyncio.run(coro)


def make_history(customer_id="CUST_TEST", account_type="Checking", rows=()):
    """Build a history from (date, payee, amount, channel, type[, timestamp]) tuples."""
    transactions = []
    for row in rows:
        date, payee, amount, channel, kind = row[:5]
        timestamp = row[5] if len(row) > 5 else None
        transactions.append(
            Transaction(
                date=date, payee=payee, amount=amount, channel=channel,
                type=kind, description="", timestamp=timestamp,
            )
        )
    return CustomerHistory(
        customer_id=customer_id, account_type=account_type, transactions=transactions
    )


def routine_rows(months=5, start_year=2024):
    """A dull, well-established history: salary in, rent and groceries out."""
    rows = []
    for month in range(1, months + 1):
        rows.append((f"{start_year}-{month:02d}-01", "Employer Ltd", 3500.00, "ACH", "credit"))
        rows.append((f"{start_year}-{month:02d}-02", "Landlord Ltd", 1200.00, "ACH", "debit"))
        for day in (5, 12, 19, 26):
            rows.append((f"{start_year}-{month:02d}-{day:02d}", "Supermarket", 80.00, "Debit Card", "debit"))
    return rows
