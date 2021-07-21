from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical, is_integer_dtype
from vivid.cacheable import cacheable


@cacheable
def read_csv(name: str, data_dir: Path):
    path = data_dir / f'{name}.csv.gz'
    return pd.read_csv(path)


def fill_na_by_unique_value(strain):
    if is_categorical(strain):
        return strain.cat.codes
    elif is_integer_dtype(strain.dtype):
        fillval = strain.min() - 1
        return strain.fillna(fillval)
    else:
        return strain.astype(str)


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(f'Mem. usage of dataframe is {start_mem:.2f} MB')
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({00 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df
