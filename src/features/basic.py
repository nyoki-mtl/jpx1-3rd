from pathlib import Path

import pandas as pd

from src.utils.utils import read_csv


def get_all_codes(stock_list):
    return stock_list.query('prediction_target')['Local Code'].unique()


def get_inputs(dataset_dir: Path):
    # 銘柄情報
    stock_list = read_csv('stock_list', dataset_dir)
    # 株価の動き
    stock_price = read_csv('stock_price', dataset_dir)
    stock_price['datetime'] = pd.to_datetime(stock_price['EndOfDayQuote Date'])
    stock_price.set_index('datetime', inplace=True)
    # 決算短信
    stock_fin = read_csv('stock_fin', dataset_dir)
    stock_fin['datetime'] = pd.to_datetime(stock_fin['base_date'])
    stock_fin.set_index('datetime', inplace=True)
    # 予測対象
    stock_labels = read_csv('stock_labels', dataset_dir)
    stock_labels['datetime'] = pd.to_datetime(stock_labels['base_date'])
    stock_labels.set_index('datetime', inplace=True)

    # dictで返す
    ret = {
        'stock_list': stock_list,
        'stock_price': stock_price,
        'stock_fin': stock_fin,
        'stock_labels': stock_labels
    }
    return ret


def get_features(stock_price, stock_fin, all_codes=None):
    cols_to_use = list(stock_price.columns.difference(stock_fin.columns)) + ['datetime', 'code']
    features = pd.merge(stock_fin, stock_price[cols_to_use], on=['datetime', 'code'], how='inner')
    # 学習時はすべて使ったほうがいいかも？
    if all_codes is not None:
        features = features[features['code'].isin(all_codes)]
    return features


def get_dataframes(features, stock_labels):
    stock_labels = stock_labels.reset_index().rename(columns={'Local Code': 'code'}).drop('base_date', axis=1)

    data = pd.merge(features, stock_labels, on=['datetime', 'code'], how='left')
    data = data[~data[['label_high_20', 'label_low_20']].isna().any(axis=1)]

    X = data.drop(['label_date_5', 'label_high_5', 'label_low_5',
                   'label_date_10', 'label_high_10', 'label_low_10',
                   'label_date_20', 'label_high_20', 'label_low_20'], axis=1).set_index('datetime')
    y = data.loc[:, ['datetime', 'label_high_20', 'label_low_20']].set_index('datetime')

    return X, y
