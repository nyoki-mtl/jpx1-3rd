import os

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import nyaggle.feature_store as fs
from nyaggle.testing import get_temp_directory


def test_save_feature():
    df = pd.DataFrame()

    df['a'] = np.arange(100)

    with get_temp_directory() as tmp:
        fs.save_feature(df, 0, tmp)

        assert os.path.exists(os.path.join(tmp, '0.f'))


def test_load_feature():
    df = pd.DataFrame()

    df['a'] = np.arange(100)

    with get_temp_directory() as tmp:
        fs.save_feature(df, 0, tmp)

        df_loaded = fs.load_feature(0, tmp)
        assert_frame_equal(df, df_loaded)


def test_multi_columns():
    df = pd.DataFrame()

    df['a'] = np.arange(100)
    df['b'] = None

    with get_temp_directory() as tmp:
        fs.save_feature(df, 0, tmp)

        df_loaded = fs.load_feature(0, tmp)
        assert_frame_equal(df, df_loaded)


def test_various_dtypes():
    df = pd.DataFrame()

    df['a'] = np.arange(100).astype(float)
    df['b'] = np.arange(100).astype(int)
    df['c'] = np.arange(100).astype(np.uint8)
    df['d'] = np.arange(100).astype(np.uint16)
    df['e'] = np.arange(100).astype(np.uint32)
    df['f'] = np.arange(100).astype(np.int8)
    df['g'] = np.arange(100).astype(np.int16)
    df['h'] = np.arange(100).astype(np.int32)
    df['i'] = np.arange(100).astype(np.int64)

    with get_temp_directory() as tmp:
        fs.save_feature(df, 0, tmp)

        df_loaded = fs.load_feature(0, tmp)
        assert_frame_equal(df, df_loaded)


def test_load_features():
    df = pd.DataFrame()

    df['a'] = np.arange(100).astype(float)
    df['b'] = np.arange(100).astype(int)
    df['c'] = np.arange(100).astype(int)

    with get_temp_directory() as tmp:
        fs.save_feature(df[['b']], 0, tmp)
        fs.save_feature(df[['c']], 1, tmp)

        df_loaded = fs.load_features(df[['a']], [0, 1], tmp)
        assert_frame_equal(df, df_loaded)


def test_load_features_no_base():
    df = pd.DataFrame()

    df['a'] = np.arange(100).astype(float)
    df['b'] = np.arange(100).astype(int)
    df['c'] = np.arange(100).astype(int)

    with get_temp_directory() as tmp:
        fs.save_feature(df[['b']], 0, tmp)
        fs.save_feature(df[['c']], 1, tmp)
        fs.save_feature(df[['a']], '2', tmp)

        df_loaded = fs.load_features(None, [0, 1, '2'], tmp)
        assert list(df_loaded.columns) == ['b', 'c', 'a']


def test_load_feature_ignore_columns():
    df = pd.DataFrame()

    df['a'] = np.arange(100).astype(float)
    df['b'] = np.arange(100).astype(int)
    df['c'] = np.arange(100).astype(int)

    with get_temp_directory() as tmp:
        fs.save_feature(df, 0, tmp)

        # just skip irrelevant column names
        df_loaded = fs.load_feature(0, tmp, ignore_columns=['b', 'X'])

        assert_frame_equal(df_loaded, df.drop('b', axis=1))


def test_load_feature_ignore_all_columns():
    df = pd.DataFrame()

    df['a'] = np.arange(100).astype(float)
    df['b'] = np.arange(100).astype(int)
    df['c'] = np.arange(100).astype(int)

    with get_temp_directory() as tmp:
        fs.save_feature(df, 0, tmp)

        df_loaded = fs.load_feature(0, tmp, ignore_columns=['a', 'b', 'c', 'X'])

        assert_frame_equal(df_loaded, df.drop(['a', 'b', 'c'], axis=1))


def test_load_features_duplicate_col_name():
    df = pd.DataFrame()

    df['a'] = np.arange(100).astype(float)
    df['b'] = np.arange(100).astype(int)
    df['c'] = np.arange(100).astype(int)

    with get_temp_directory() as tmp:
        fs.save_feature(df[['a', 'b']], 0, tmp)
        fs.save_feature(df[['b', 'c']], 1, tmp)
        fs.save_feature(df[['b', 'a']], 'X', tmp)

        df_loaded = fs.load_features(None, [0, 1, 'X'], tmp, rename_duplicate=True)
        assert list(df_loaded.columns) == ['a', 'b', 'b_1', 'c', 'b_X', 'a_X']

        df_loaded = fs.load_features(None, [0, 1, 'X'], tmp, rename_duplicate=False)
        assert list(df_loaded.columns) == ['a', 'b', 'b', 'c', 'b', 'a']


def test_invalid_feature():
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 5] + [None] * 5,
        'b': np.random.randint(0, 10, size=10)
    })
    y = pd.Series([1, 0, 1, 0, 1])

    with get_temp_directory() as tmp:
        with pytest.raises(RuntimeError):
            fs.save_feature(df[['a']], 0, reference_target_variable=y, directory=tmp)
        with pytest.raises(RuntimeError):
            fs.save_feature(df, 0, reference_target_variable=y, directory=tmp)

        # ok
        fs.save_feature(df[['b']], 0, reference_target_variable=y, directory=tmp)


def test_feature_exists():
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 5] + [None] * 5
    })

    with get_temp_directory() as tmp:
        fs.save_feature(df[['a']], 0, directory=tmp)
        with pytest.raises(RuntimeError):
            fs.save_feature(df, 0, overwrite=False, directory=tmp)


def test_decorator():
    with get_temp_directory() as tmp:
        @fs.cached_feature('x', tmp)
        def make_feature_x():
            return pd.DataFrame({'a': [1, 2, 3, 4, 5]})

        @fs.cached_feature('y', tmp)
        def make_feature_y(n: int):
            return pd.DataFrame({'b': np.arange(n)})

        x = make_feature_x()
        assert make_feature_x.__name__ == "make_feature_x"
        assert os.path.exists(os.path.join(tmp, "x.f"))
        x2 = make_feature_x()
        assert_frame_equal(x, x2)

        y = make_feature_y(100)
        assert len(y) == 100
        assert os.path.exists(os.path.join(tmp, "y.f"))
        y2 = make_feature_y(100)
        assert_frame_equal(y, y2)
