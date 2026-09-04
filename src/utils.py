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
