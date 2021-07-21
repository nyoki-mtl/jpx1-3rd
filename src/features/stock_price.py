import numpy as np
import pandas as pd


def get_price_limit(value):
    if value < 100:
        return 30
    elif value < 200:
        return 50
    elif value < 500:
        return 80
    elif value < 700:
        return 100
    elif value < 1000:
        return 150
    elif value < 1500:
        return 300
    elif value < 2000:
        return 400
    elif value < 3000:
        return 500
    elif value < 5000:
        return 700
    elif value < 7000:
        return 1000
    elif value < 10_000:
        return 1500
    elif value < 15_000:
        return 3000
    elif value < 20_000:
        return 4000
    elif value < 30_000:
        return 5000
    elif value < 50_000:
        return 7000
    elif value < 70_000:
        return 10_000
    elif value < 100_000:
        return 15_000
    elif value < 150_000:
        return 30_000
    elif value < 200_000:
        return 40_000
    elif value < 300_000:
        return 50_000
    elif value < 500_000:
        return 70_000
    elif value < 700_000:
        return 100_000
    elif value < 1_000_000:
        return 150_000
    elif value < 1_500_000:
        return 300_000
    elif value < 2_000_000:
        return 400_000
    elif value < 3_000_000:
        return 500_000
    elif value < 5_000_000:
        return 700_000
    elif value < 7_000_000:
        return 1_000_000
    elif value < 10_000_000:
        return 1_500_000
    elif value < 15_000_000:
        return 3_000_000
    elif value < 20_000_000:
        return 4_000_000
    elif value < 30_000_000:
        return 5_000_000
    elif value < 50_000_000:
        return 7_000_000
    else:
        return 10_000_000


def make_stock_price_features(stock_price, stock_list, nikkei225):
    date_list = [5, 10, 20, 40, 60, 80, 120, 240]
    target_columns = [
        'Local Code', 'EndOfDayQuote Open', 'EndOfDayQuote High', 'EndOfDayQuote Low',
        'EndOfDayQuote ExchangeOfficialClose', 'EndOfDayQuote CumulativeAdjustmentFactor',
        'EndOfDayQuote Volume', 'EndOfDayQuote VWAP'
    ]
    stock_price = stock_price.loc[:, target_columns].rename(columns={
        'Local Code': 'code',
        'EndOfDayQuote ExchangeOfficialClose': 'official close',
        'EndOfDayQuote CumulativeAdjustmentFactor': 'cum adj factor',
        'EndOfDayQuote Open': 'open',
        'EndOfDayQuote High': 'high',
        'EndOfDayQuote Low': 'low',
        'EndOfDayQuote Volume': 'volume',
        'EndOfDayQuote VWAP': 'vwap',
    })

    drop_columns = [
        'nikkei225'
    ]

    stock_price['adj_eo_close'] = stock_price['official close'] * stock_price['cum adj factor']
    stock_price['adj_eo_high'] = stock_price['high'] * stock_price['cum adj factor']
    stock_price['adj_eo_low'] = stock_price['low'] * stock_price['cum adj factor']
    stock_price['is_traded'] = stock_price['volume'] > 0

    stock_price['price_limit'] = stock_price['adj_eo_close'].apply(get_price_limit)
    stock_price['upper_pct'] = (stock_price['adj_eo_close'] + stock_price['price_limit']) / stock_price[
        'adj_eo_close'] - 1
    stock_price['lower_pct'] = (stock_price['adj_eo_close'] - stock_price['price_limit']) / stock_price[
        'adj_eo_close'] - 1

    target_columns = ['code', 'open', 'high', 'low', 'official close', 'volume', 'vwap',
                      'adj_eo_close', 'adj_eo_high', 'adj_eo_low', 'is_traded', 'price_limit',
                      'upper_pct', 'lower_pct']
    stock_price = stock_price.loc[:, target_columns].reset_index().sort_values(['code', 'datetime'])

    # 日経225
    stock_list = stock_list.rename(columns={'Local Code': 'code'})
    stock_list['nikkei225'] = stock_list['code'].isin(nikkei225)
    stock_price = pd.merge(stock_price, stock_list, on='code', how='left')

    g = stock_price.groupby('code')['official close']

    for d in [1, 2, 3, 4] + date_list:
        # 終値のn営業日リターン
        stock_price[f'return_{d}day'] = g.pct_change(d)

    for d in date_list:
        # 終値のn営業日ボラティリティ
        stock_price[f'volatility_{d}day'] = g.apply(np.log).diff().rolling(d).std().values
        # applyでnp.logをするとnanになってほしい箇所がnanにならない
        stock_price[f'valid{d}'] = True
        stock_price.loc[stock_price.groupby('code').head(d).index, f'valid{d}'] = False
        stock_price.loc[~stock_price[f'valid{d}'], f'volatility_{d}day'] = np.nan
        drop_columns.append(f'valid{d}')

        # 終値とn営業日の単純移動平均線の乖離
        stock_price[f'MA_gap_{d}day'] = stock_price['official close'] / g.rolling(d).mean().values

        # 直近n営業日の最高値/最安値
        stock_price[f'high_value_{d}day'] = stock_price.groupby('code')['high'].rolling(d).max().values
        stock_price[f'low_value_{d}day'] = stock_price.groupby('code')['low'].rolling(d).min().values
        stock_price[f'high_value_{d}day'] = stock_price[f'high_value_{d}day'] / stock_price['official close'] - 1
        stock_price[f'low_value_{d}day'] = stock_price[f'low_value_{d}day'] / stock_price['official close'] - 1

        # 取引量における直近n営業日の単純移動平均線の乖離
        stock_price[f'volume_gap_{d}day'] = stock_price.groupby('code')['volume'].rolling(d).mean().values
        stock_price[f'volume_gap_{d}day'] = stock_price['volume'] / stock_price[f'volume_gap_{d}day']

    # 制限値幅
    tmp = stock_price.groupby('code')[['adj_eo_close', 'price_limit']].shift(1)
    stock_price['limit_high'] = tmp['adj_eo_close'] + tmp['price_limit']
    stock_price['limit_low'] = tmp['adj_eo_close'] - tmp['price_limit']

    # # ストップ高/安だったかどうか
    # stock_price['is_limit_high'] = stock_price['limit_high'] == stock_price['adj_eo_high']
    # stock_price['is_limit_low'] = stock_price['limit_low'] == stock_price['adj_eo_low']
    # stock_price['is_limit'] = stock_price['is_limit_high'] & stock_price['is_limit_low']
    #
    # gh = stock_price.groupby('code')['is_limit_high']
    # gl = stock_price.groupby('code')['is_limit_low']
    # ga = stock_price.groupby('code')['is_limit']
    # for d in date_list:
    #     # 直近n営業日にどれだけストップ高/安があったか
    #     stock_price[f'is_limit_high_{d}day'] = gh.rolling(d).mean().values
    #     stock_price[f'is_limit_low_{d}day'] = gl.rolling(d).mean().values
    #     stock_price[f'is_limit_{d}day'] = ga.rolling(d).mean().values

    # 値の正規化
    stock_price['open'] = stock_price['open'] / stock_price['official close'] - 1
    stock_price['high'] = stock_price['high'] / stock_price['official close'] - 1
    stock_price['low'] = stock_price['low'] / stock_price['official close'] - 1
    stock_price['vwap'] = stock_price['vwap'] / stock_price['official close'] - 1

    # 33 sector code
    for c in ['33 Sector(Code)', '17 Sector(Code)', 'Size Code (New Index Series)']:
        mean_sector = stock_price.groupby(['datetime', c])['official close'].mean().reset_index()
        g = mean_sector.groupby(c)['official close']
        for d in [1, 2, 3, 4] + date_list:
            mean_sector[c + f'return_{d}day'] = g.pct_change(d)
        mean_sector.drop(['official close'], axis=1, inplace=True)

        stock_price = pd.merge(stock_price, mean_sector, on=['datetime', c], how='left')

    # 日経225
    mean_nikkei225 = stock_price[stock_price['nikkei225']].groupby(['datetime'])['official close'].mean()
    mean_nikkei225 = mean_nikkei225.reset_index().set_index('datetime').sort_index()
    target = mean_nikkei225['official close']

    for d in [1, 2, 3, 4] + date_list:
        mean_nikkei225[f'nikkei225_return_{d}day'] = target.pct_change(d)

    for d in date_list:
        mean_nikkei225[f'nikkei225_volatility_{d}day'] = target.apply(np.log).diff().rolling(d).std().values
        mean_nikkei225[f'nikkei225_MA_gap_{d}day'] = target / target.rolling(d).mean().values
        mean_nikkei225[f'nikkei225_high_value_{d}day'] = target / target.rolling(d).max().values
        mean_nikkei225[f'nikkei225_low_value_{d}day'] = target / target.rolling(d).min().values

    mean_nikkei225 = mean_nikkei225.drop('official close', axis=1).reset_index()
    stock_price = pd.merge(stock_price, mean_nikkei225, on='datetime', how='left')

    stock_price.drop(drop_columns, axis=1, inplace=True)

    return stock_price


if __name__ == '__main__':
    from pathlib import Path
    from src.features.basic import get_inputs

    data_dir = Path('./data/jpx/')
    dfs = get_inputs(data_dir)

    stock_list = dfs['stock_list']
    stock_price = dfs['stock_price']
    stock_fin = dfs['stock_fin']
    stock_labels = dfs['stock_labels']

    with open('./tools/infos/nikkei225.txt', 'r') as f:
        nikkei225 = f.readlines()
    nikkei225 = sorted([int(s.strip()) for s in nikkei225])

    stock_price_features = make_stock_price_features(stock_price, stock_list, nikkei225)
