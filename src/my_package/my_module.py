"""Generic example module"""
from string import ascii_uppercase
import numpy as np
import pandas as pd

def make_dataframe(ncols: int = 4, nrows: int = 5) -> pd.DataFrame:
    """
    Generate a pandas.DataFrame of random data with ncols columns and
    nrows index length.

    Parameters
    ----------
    ncols: int, optional, default 4
        Number of columns for resulting dataframe.
    nrows: int, optional, default 5
        Number of rows of resulting dataframe.

    Returns
    -------
    pandas.DataFrame with ncols columns and index length nrows.
    
    """
    # Return dataframe
    return pd.DataFrame({c: np.random.random(nrows) for c in ascii_uppercase[:ncols]})
