from datetime import datetime
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
# from nyaggle.validation import TimeSeriesSplit
from sklearn.model_selection import BaseCrossValidator


def get_ts1(X_train, vidx):
    ts = VersionAwareTimeSeriesSplit(X_train.index.to_series(), vidx)
    ts.add_fold(train_interval=('2016-01-01', '2017-05-31'), test_interval=('2017-07-01', '2018-06-30'))
    ts.add_fold(train_interval=('2016-01-01', '2017-11-30'), test_interval=('2018-01-01', '2018-12-31'))
    ts.add_fold(train_interval=('2016-01-01', '2018-05-31'), test_interval=('2018-07-01', '2019-06-30'))
    ts.add_fold(train_interval=('2016-01-01', '2018-11-30'), test_interval=('2019-01-01', '2019-12-31'))
    return ts


def get_ts2(X_train, vidx):
    ts = VersionAwareTimeSeriesSplit(X_train.index.to_series(), vidx)
    ts.add_fold(train_interval=('2016-01-01', '2018-11-30'), test_interval=('2019-01-01', '2019-12-31'))
    ts.add_fold(train_interval=('2016-01-01', '2019-05-31'), test_interval=('2019-07-01', '2020-06-30'))
    ts.add_fold(train_interval=('2016-01-01', '2019-11-30'), test_interval=('2020-01-01', '2020-12-31'))
    ts.add_fold(train_interval=('2016-01-01', '2020-02-28'), test_interval=('2020-04-01', '2021-03-31'))
    return ts


class VersionAwareTimeSeriesSplit(BaseCrossValidator):
    datepair = Tuple[Union[datetime, str], Union[datetime, str]]

    def __init__(self, source: Union[pd.Series, str], vidx: np.ndarray,
                 times: List[Tuple[datepair, datepair]] = None):
        super().__init__()
        self.source = source
        self.vidx = vidx
        self.times = []
        if times:
            for t in times:
                self.add_fold(t[0], t[1])

    def _to_datetime(self, time: Union[str, datetime]):
        return time if isinstance(time, datetime) else pd.to_datetime(time)

    def _to_datetime_tuple(self, time: datepair):
        return self._to_datetime(time[0]), self._to_datetime(time[1])

    def add_fold(self, train_interval: datepair, test_interval: datepair):
        """
        Append 1 split to the validator.

        Args:
            train_interval:
                start and end time of training data.
            test_interval:
                start and end time of test data.
        """
        train_interval = self._to_datetime_tuple(train_interval)
        test_interval = self._to_datetime_tuple(test_interval)
        assert train_interval[1], "train_interval[1] should not be None"
        assert test_interval[0], "test_interval[0] should not be None"

        assert (not train_interval[0]) or (
                train_interval[0] <= train_interval[1]), "train_interval[0] < train_interval[1]"
        assert (not test_interval[1]) or (test_interval[0] <= test_interval[1]), "test_interval[0] < test_interval[1]"

        self.times.append((train_interval, test_interval))

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.times)

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Args:
            X:
                Training data.
            y:
                Ignored.
            groups:
                Ignored.

        Yields:
            The training set and the testing set indices for that split.
        """
        ts = X[self.source] if isinstance(self.source, str) else self.source
        assert len(ts) == len(self.vidx)

        for train_interval, test_interval in self.times:
            train_mask = ts < train_interval[1]
            if train_interval[0]:
                train_mask = (train_interval[0] <= ts) & train_mask

            test_mask = test_interval[0] <= ts
            if test_interval[1]:
                test_mask = test_mask & (ts < test_interval[1])

            test_mask = test_mask & self.vidx

            yield np.where(train_mask)[0], np.where(test_mask)[0]
