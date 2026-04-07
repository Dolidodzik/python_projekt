def format_float_to_string(value):
    """Format a float to 6 decimal places as a string (no scientific notation)."""
    if value is None:
        return None
    return f"{value:.6f}"


def adjusted_r_squared_from_r2(r2: float, n: int, p: int):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)
