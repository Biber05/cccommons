import functools
import time
from typing import Union

import numpy as np
import pandas as pd


def product(lists: [list]) -> list:
    """

    Args:
        lists:

    Returns:

    """
    import itertools

    return list(itertools.product(*lists))


def log(data: Union[list, dict, np.ndarray]):
    if isinstance(data, list):
        [print(x) for x in data]
    elif isinstance(data, dict):
        for k, v in data.items():
            print("{} : {}".format(k, v))
    else:
        with np.printoptions(precision=3, suppress=True):
            print(data)


def str_to_int(string: str) -> int:
    return int(float(string))


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


def json_to_df(json) -> pd.DataFrame:
    """

    Args:
        json:

    Returns:
        object: Pandas DataFrame

    """
    return pd.read_json(json)


def dict_to_df(dictionary: dict) -> pd.DataFrame:
    """

    Args:
        dictionary:

    Returns:

    """
    return pd.DataFrame.from_dict(data=dictionary, orient="index").fillna(0)


def list_to_df(array: [list]) -> pd.DataFrame:
    """

    Args:
        array: list of lists

    Returns:
        pandas DataFrame
    """
    return pd.DataFrame(array).fillna(0)


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
