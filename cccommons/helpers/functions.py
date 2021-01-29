import functools
import time


def product(lists: [list]) -> list:
    """

    Args:
        lists:

    Returns:

    """
    import itertools

    return list(itertools.product(*lists))


def timer(func):
    """
    Print the runtime of the decorated function
    """
    import logging

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        logging.getLogger(func.__name__).info(
            f"Finished {func.__name__!r} in {run_time:.4f} secs"
        )
        return value

    return wrapper_timer


def expand_grid(dictionary: dict) -> list:
    """
    Args:
        dictionary:

    Returns:
        object:

    """
    from itertools import product

    return [row for row in product(*dictionary.values())]


def name(func: str, param) -> str:
    return f"{func}_{param}.tiff"


__all__ = [
    "product",
    "log",
    "str_to_int",
    "list_to_df",
    "dict_to_df",
    "name",
    "json_to_df",
    "expand_grid",
]
