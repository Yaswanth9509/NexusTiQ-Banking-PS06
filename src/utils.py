"""Utility Functions - Helper methods for analysis"""


def calculate_median(values: list) -> float:
    """Calculate median of a list of values"""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n % 2 == 1:
        return float(sorted_values[n // 2])
    else:
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2


def calculate_std_dev(values: list) -> float:
    """Calculate standard deviation"""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def make_transaction_id(customer_id: str, index: int, date: str, payee: str, amount: float) -> str:
    """
    Derive a stable identifier for a transaction.

    A random id would be regenerated on every parse, so the same transaction
    would carry a different reference in every report and nothing could be
    followed back to the history it came from. Deriving it from the content
    plus the row's position makes it reproducible, and unique even when a
    customer pays the same payee the same amount twice in one day.
    """
    import hashlib

    seed = f"{customer_id}|{index}|{date}|{payee}|{amount:.2f}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8].upper()
    return f"TXN_{digest}"
