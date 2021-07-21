import sys

import numpy.testing as npt
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from nyaggle.feature.nlp import BertSentenceVectorizer

_TEST_SENTENCE_EN = [
    'This is a pen.',
    'A quick brown fox',
    'Redistribution and use in source and binary forms, with or without modification.',
    'BERT is the state of the art NLP model.',
    'This is a pen.',
    'THIS IS A PEN.',
]

_TEST_SENTENCE_JP = [
    '金メダルが5枚欲しい。',
    '私は昨日から風邪をひいています。',
    'これはペンです。',
    'BERTは最新の自然言語処理モデルです。',
    '金メダルが5枚欲しい。',
    '金メダルが 5枚 欲しい。',
]


def _under_py35():
    return not (sys.version_info.major == 3 and sys.version_info.minor >= 6)


@pytest.mark.skipif(_under_py35(), reason="BertSentenceVectorizer is not supported under Python <= 3.5")
def test_bert_fit():
    bert = BertSentenceVectorizer(use_cuda=False)

    X = pd.DataFrame({
        'id': [0, 1, 2, 3, 4, 5],
        'sentence': _TEST_SENTENCE_EN
    })

    bert.fit(X)
    ret = bert.transform(X)

    assert ret.shape[0] == 6
    assert ret.shape[1] == 768 + 1  # id + embed

    ret.drop('id', axis=1, inplace=True)
    npt.assert_almost_equal(ret.iloc[0, :].values, ret.iloc[4, :].values)
    npt.assert_almost_equal(ret.iloc[0, :].values, ret.iloc[5, :].values)


@pytest.mark.skipif(_under_py35(), reason="BertSentenceVectorizer is not supported under Python <= 3.5")
def test_bert_fit_transform():
    X = pd.DataFrame({
        'id': [0, 1, 2, 3, 4, 5],
        'sentence': _TEST_SENTENCE_EN
    })

    bert = BertSentenceVectorizer(use_cuda=False)
    ret = bert.fit_transform(X)

    bert = BertSentenceVectorizer(use_cuda=False)
    bert.fit(X)
    ret2 = bert.fit_transform(X)

    assert_frame_equal(ret, ret2)


@pytest.mark.skipif(_under_py35(), reason="BertSentenceVectorizer is not supported under Python <= 3.5")
def test_bert_en_svd():
    n_components = 3
    bert = BertSentenceVectorizer(n_components=n_components, use_cuda=False)

    X = pd.DataFrame({
        'id': [0, 1, 2, 3, 4, 5],
        'sentence': _TEST_SENTENCE_EN
    })

    ret = bert.fit_transform(X)

    assert ret.shape[0] == 6
    assert ret.shape[1] == n_components + 1

    ret.drop('id', axis=1, inplace=True)
    npt.assert_almost_equal(ret.iloc[0, :].values, ret.iloc[4, :].values, decimal=3)
    npt.assert_almost_equal(ret.iloc[0, :].values, ret.iloc[5, :].values, decimal=3)


@pytest.mark.skipif(_under_py35(), reason="BertSentenceVectorizer is not supported under Python <= 3.5")
def test_bert_en_svd_multicol():
    bert = BertSentenceVectorizer(use_cuda=False)

    X = pd.DataFrame({
        'id': [0, 1, 2, 3, 4, 5],
        'sentence': _TEST_SENTENCE_EN,
        'sentence2': _TEST_SENTENCE_EN
    })

    ret = bert.fit_transform(X)

    assert ret.shape[0] == 6
    assert ret.shape[1] == 2 * 768 + 1

    ret.drop('id', axis=1, inplace=True)
    npt.assert_almost_equal(ret.iloc[0, :].values, ret.iloc[4, :].values, decimal=3)
    npt.assert_almost_equal(ret.iloc[0, :].values, ret.iloc[5, :].values, decimal=3)


@pytest.mark.skipif(_under_py35(), reason="BertSentenceVectorizer is not supported under Python <= 3.5")
def test_bert_jp():
    bert = BertSentenceVectorizer(use_cuda=False, lang='jp')

    X = pd.DataFrame({
        'id': [0, 1, 2, 3, 4, 5],
        'sentence': _TEST_SENTENCE_JP
    })

    ret = bert.fit_transform(X)

    assert ret.shape[0] == 6
    assert ret.shape[1] == 768 + 1

    ret.drop('id', axis=1, inplace=True)
    npt.assert_almost_equal(ret.iloc[0, :].values, ret.iloc[4, :].values)
    npt.assert_almost_equal(ret.iloc[0, :].values, ret.iloc[5, :].values)
