# -*- coding: utf-8 -*-
import io
import os
import gc
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical, is_integer_dtype
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder


def signate_metric(y_true, y_pred):
    high_score = spearmanr(y_true['label_high_20'].values, y_pred['label_high_20'])[0]
    low_score = spearmanr(y_true['label_low_20'].values, y_pred['label_low_20'])[0]
    return (high_score - 1) ** 2 + (low_score - 1) ** 2, high_score, low_score


rename_dict = {
    'Local Code': 'code',
    'Result_FinancialStatement AccountingStandard': 'RF AccountingStandard',
    'Result_FinancialStatement FiscalPeriodEnd': 'RF FiscalPeriodEnd',
    'Result_FinancialStatement ReportType': 'RF ReportType',
    'Result_FinancialStatement FiscalYear': 'RF FiscalYear',
    'Result_FinancialStatement ModifyDate': 'RF ModifyDate',
    'Result_FinancialStatement CompanyType': 'RF CompanyType',
    'Result_FinancialStatement ChangeOfFiscalYearEnd': 'RF ChangeOfFiscalYearEnd',
    'Result_FinancialStatement NetSales': 'RF NetSales',
    'Result_FinancialStatement OperatingIncome': 'RF OperatingIncome',
    'Result_FinancialStatement OrdinaryIncome': 'RF OrdinaryIncome',
    'Result_FinancialStatement NetIncome': 'RF NetIncome',
    'Result_FinancialStatement TotalAssets': 'RF TotalAssets',
    'Result_FinancialStatement NetAssets': 'RF NetAssets',
    'Result_FinancialStatement CashFlowsFromOperatingActivities': 'RF CashFlowsFromOperating',
    'Result_FinancialStatement CashFlowsFromFinancingActivities': 'RF CashFlowsFromFinancing',
    'Result_FinancialStatement CashFlowsFromInvestingActivities': 'RF CashFlowsFromInvesting',
    'Forecast_FinancialStatement AccountingStandard': 'FF AccountingStandard',
    'Forecast_FinancialStatement FiscalPeriodEnd': 'FF FiscalPeriodEnd',
    'Forecast_FinancialStatement ReportType': 'FF ReportType',
    'Forecast_FinancialStatement FiscalYear': 'FF FiscalYear',
    'Forecast_FinancialStatement ModifyDate': 'FF ModifyDate',
    'Forecast_FinancialStatement CompanyType': 'FF CompanyType',
    'Forecast_FinancialStatement ChangeOfFiscalYearEnd': 'FF ChangeOfFiscalYearEnd',
    'Forecast_FinancialStatement NetSales': 'FF NetSales',
    'Forecast_FinancialStatement OperatingIncome': 'FF OperatingIncome',
    'Forecast_FinancialStatement OrdinaryIncome': 'FF OrdinaryIncome',
    'Forecast_FinancialStatement NetIncome': 'FF NetIncome',
    'Result_Dividend FiscalPeriodEnd': 'RD FiscalPeriodEnd',
    'Result_Dividend ReportType': 'RD ReportType',
    'Result_Dividend FiscalYear': 'RD FiscalYear',
    'Result_Dividend ModifyDate': 'RD ModifyDate',
    'Result_Dividend RecordDate': 'RD RecordDate',
    'Result_Dividend DividendPayableDate': 'RD DividendPayableDate',
    'Result_Dividend QuarterlyDividendPerShare': 'RD QuarterlyDividendPerShare',
    'Result_Dividend AnnualDividendPerShare': 'RD AnnualDividendPerShare',
    'Forecast_Dividend FiscalPeriodEnd': 'FD FiscalPeriodEnd',
    'Forecast_Dividend ReportType': 'FD ReportType',
    'Forecast_Dividend FiscalYear': 'FD FiscalYear',
    'Forecast_Dividend ModifyDate': 'FD ModifyDate',
    'Forecast_Dividend RecordDate': 'FD RecordDate',
    'Forecast_Dividend QuarterlyDividendPerShare': 'FD QuarterlyDividendPerShare',
    'Forecast_Dividend AnnualDividendPerShare': 'FD AnnualDividendPerShare'
}

drop_columns = [
    'base_date',
    'RF FiscalPeriodEnd',
    'RF FiscalYear',
    'RF ModifyDate',
    'RF ChangeOfFiscalYearEnd',
    'FF AccountingStandard',
    'FF FiscalPeriodEnd',
    'FF ReportType',
    'FF FiscalYear',
    'FF ModifyDate',
    'FF CompanyType',
    'FF ChangeOfFiscalYearEnd',
    'FF NetSales',
    'FF OperatingIncome',
    'FF OrdinaryIncome',
    'FF NetIncome',
    'RD FiscalPeriodEnd',
    'RD ReportType',
    'RD FiscalYear',
    'RD ModifyDate',
    'RD RecordDate',
    'RD DividendPayableDate',
    'RD QuarterlyDividendPerShare',
    'RD AnnualDividendPerShare',
    'FD FiscalPeriodEnd',
    'FD ReportType',
    'FD FiscalYear',
    'FD ModifyDate',
    'FD RecordDate',
    'FD QuarterlyDividendPerShare',
    'FD AnnualDividendPerShare',

    'prediction_target',
    'Effective Date',
    'Name (English)',
    'Section/Products',
    '33 Sector(Code)',
    '33 Sector(name)',
    '17 Sector(Code)',
    '17 Sector(name)',
    'Size Code (New Index Series)',
    'Size (New Index Series)',
    'IssuedShareEquityQuote AccountingStandard',
    'IssuedShareEquityQuote ModifyDate',
    'IssuedShareEquityQuote IssuedShare',
    'official close',

    'RF AccountingStandard',
    'is_traded',
    'first_repot',
    'price_limit',
    'version',
]


def get_version_on_stock_fin(stock_fin, latest_version_idx=False):
    groupby_columns = ['code', 'RF ReportType', 'RF FiscalYear']
    tmp_df = stock_fin.reset_index()[groupby_columns].copy()
    tmp_df['cnt'] = 1
    tmp_df['version'] = tmp_df.sort_index().groupby(groupby_columns)['cnt'].cumsum()
    nan_idx = tmp_df['RF ReportType'].isna() | tmp_df[
        'RF FiscalYear'].isna()
    tmp_df.loc[nan_idx, 'version'] = 1
    tmp_df['RF ReportType'].fillna('NaN')
    if latest_version_idx:
        latest = np.zeros(len(tmp_df), dtype='bool')
        latest[tmp_df.reset_index().groupby(groupby_columns)['version'].idxmax().values] = True
        return tmp_df['version'], latest
    else:
        return tmp_df['version']


def make_stock_fin_features(stock_fin, stock_list, stock_price, normalize_value=None):
    stock_fin = pd.merge(stock_fin.reset_index(), stock_list, how='left', on='Local Code')
    stock_fin.rename(columns=rename_dict, inplace=True)
    eod_close = stock_price.reset_index()[['datetime', 'Local Code', 'EndOfDayQuote ExchangeOfficialClose']]
    eod_close.rename(columns={
        'EndOfDayQuote ExchangeOfficialClose': 'official close',
        'Local Code': 'code'}, inplace=True)
    stock_fin = pd.merge(stock_fin, eod_close, on=['datetime', 'code'], how='left')

    # 訂正報が来ている場合、過去のレポートは未来情報を見ることになるが、最新版だけ予想するルールなのでOK
    version, latest = get_version_on_stock_fin(stock_fin, True)
    stock_fin['version'] = version
    stock_fin['latest'] = latest
    stock_fin['first_repot'] = stock_fin['version'] == 1

    # 配当がすべてnanの会社は四半期配当で埋める
    tmp = stock_fin.groupby('code')['RD AnnualDividendPerShare'].mean()
    idx = stock_fin['code'].isin(tmp[tmp.isna()].index.values)
    stock_fin.loc[idx, 'RD AnnualDividendPerShare'] = stock_fin.loc[idx, 'RD QuarterlyDividendPerShare']
    # それでもすべてnanなら0埋め
    tmp = stock_fin.groupby('code')['RD AnnualDividendPerShare'].mean()
    idx = stock_fin['code'].isin(tmp[tmp.isna()].index.values)
    stock_fin.loc[idx, 'RD AnnualDividendPerShare'] = stock_fin.loc[idx, 'RD AnnualDividendPerShare'].fillna(0)
    # ときどきnanがある会社なら前方の値を使って穴埋め
    stock_fin['RD AnnualDividendPerShare'] = stock_fin.groupby('code')['RD AnnualDividendPerShare'].fillna(
        method='ffill')

    # 配当がすべてnanの会社は四半期配当で埋める
    tmp = stock_fin.groupby('code')['FD AnnualDividendPerShare'].mean()
    idx = stock_fin['code'].isin(tmp[tmp.isna()].index.values)
    stock_fin.loc[idx, 'FD AnnualDividendPerShare'] = stock_fin.loc[idx, 'FD QuarterlyDividendPerShare']
    # それでもすべてnanなら0埋め
    tmp = stock_fin.groupby('code')['FD AnnualDividendPerShare'].mean()
    idx = stock_fin['code'].isin(tmp[tmp.isna()].index.values)
    stock_fin.loc[idx, 'FD AnnualDividendPerShare'] = stock_fin.loc[idx, 'FD AnnualDividendPerShare'].fillna(0)
    # ときどきnanがある会社なら前方の値を使って穴埋め
    stock_fin['FD AnnualDividendPerShare'] = stock_fin.groupby('code')['FD AnnualDividendPerShare'].fillna(
        method='ffill')

    # 財務指標
    stock_fin['market_cap'] = stock_fin['official close'] * stock_fin['IssuedShareEquityQuote IssuedShare']  # 時価総額
    stock_fin['PER'] = stock_fin['official close'] / (stock_fin['RF NetIncome'] * 1_000_000 / stock_fin[
        'IssuedShareEquityQuote IssuedShare'])  # PER(株価収益率)
    stock_fin.loc[stock_fin['RF CashFlowsFromOperating'] == 0, 'PER'] = np.nan  # 営業キャッシュフロー
    stock_fin['PBR'] = stock_fin['official close'] / (stock_fin['RF NetAssets'] * 1_000_000 / stock_fin[
        'IssuedShareEquityQuote IssuedShare'])  # PBR(株価純資産倍率)
    stock_fin['ROE'] = stock_fin['PBR'] / stock_fin['PER']  # ROE(自己資本利益率)
    stock_fin['profit_margin'] = stock_fin['RF NetIncome'] / stock_fin['RF NetSales']  # 純利益率
    stock_fin.loc[stock_fin['RF CashFlowsFromOperating'] == 0, 'profit_margin'] = np.nan
    stock_fin['equity_ratio'] = stock_fin['RF NetAssets'] / stock_fin['RF TotalAssets']  # 自己資本比率
    stock_fin['dividend_yield'] = stock_fin['RD AnnualDividendPerShare'] / stock_fin[
        'official close'] * 100  # 配当利回り
    stock_fin['sum_cache_flow'] = stock_fin['RF CashFlowsFromOperating'] + \
                                  stock_fin['RF CashFlowsFromFinancing'] + \
                                  stock_fin['RF CashFlowsFromInvesting']

    stock_fin['ForecastPER'] = stock_fin['official close'] / (stock_fin['FF NetIncome'] * 1_000_000 / stock_fin[
        'IssuedShareEquityQuote IssuedShare'])  # 来期PER
    stock_fin.loc[stock_fin['RF CashFlowsFromOperating'] == 0, 'ForecastPER'] = np.nan
    stock_fin['Forecastprofit_margin'] = stock_fin['FF NetIncome'] / stock_fin['FF NetSales']  # 来期純利益率
    stock_fin.loc[stock_fin['RF CashFlowsFromOperating'] == 0, 'Forecastprofit_margin'] = np.nan
    stock_fin['Forecastdividend_yield'] = stock_fin['FD AnnualDividendPerShare'] / stock_fin[
        'official close'] * 100  # 来期配当利回り

    # 配当基準日が今後21日にあるか(=権利落ち日が20日以内にあるか), n日以内にあるか
    timedelta = pd.to_datetime(stock_fin['FD RecordDate']) - stock_fin['datetime']
    stock_fin['RecordDateIn21Days'] = timedelta <= pd.Timedelta(21, 'D')
    stock_fin['RecordDateIn41Days'] = timedelta <= pd.Timedelta(41, 'D')
    stock_fin['RecordDateIn61Days'] = timedelta <= pd.Timedelta(61, 'D')

    stock_fin.loc[
        stock_fin['FD RecordDate'].isna(), ['RecordDateIn21Days', 'RecordDateIn41Days', 'RecordDateIn61Days']] = np.nan

    # 前の決算からの予想のずれ
    forecast_columns = ['FF NetSales',
                        'FF OperatingIncome',
                        'FF OrdinaryIncome',
                        'FF NetIncome',
                        'ForecastPER',
                        'Forecastprofit_margin',
                        'Forecastdividend_yield']
    result_columns = ['RF NetSales',
                      'RF OperatingIncome',
                      'RF OrdinaryIncome',
                      'RF NetIncome',
                      'PER',
                      'profit_margin',
                      'dividend_yield']
    assert len(forecast_columns) == len(result_columns)
    previous_forecast_columns = ['Previous' + c for c in forecast_columns]

    prev_forecast = stock_fin[latest].groupby('code')[forecast_columns].shift(1)
    for prev_forecast_c in previous_forecast_columns:
        stock_fin[prev_forecast_c] = np.nan

    stock_fin.loc[stock_fin.index.isin(prev_forecast.index), previous_forecast_columns] = prev_forecast.values
    for prev_forecast_c, result_c in zip(previous_forecast_columns, result_columns):
        stock_fin[result_c + 'pct'] = stock_fin[result_c] / stock_fin[prev_forecast_c] - 1

    # 正規化
    normalize_target = ['profit_margin', 'PER', 'PBR', 'ROE', 'equity_ratio', 'sum_cache_flow',
                        'RF NetSales', 'RF OperatingIncome', 'RF OrdinaryIncome', 'RF NetIncome', 'RF TotalAssets',
                        'RF NetAssets',
                        'RF CashFlowsFromOperating', 'RF CashFlowsFromFinancing', 'RF CashFlowsFromInvesting']
    norm_rename_dict = {c: 'norm_' + c for c in normalize_target}
    if normalize_value is not None:
        normalize_value = stock_fin[stock_fin.datetime <= '2021-03-24'].groupby(['33 Sector(Code)'])[
            normalize_target].mean().reset_index()
    normalize_value = normalize_value.rename(columns=norm_rename_dict)
    stock_fin = pd.merge(stock_fin, normalize_value, on=['33 Sector(Code)'], how='left')
    norm_rename_dict.update({'Forecastprofit_margin': 'norm_profit_margin'})
    for c, nc in norm_rename_dict.items():
        stock_fin[c + '_33normalized'] = stock_fin[c] / stock_fin[nc]
    stock_fin.drop(list(norm_rename_dict.values()), axis=1, inplace=True)

    # pctをとる
    for c in ['PER', 'PBR', 'ROE', 'profit_margin', 'equity_ratio', 'sum_cache_flow',
              'RF NetSales',  # 売上高（単位：百万円）
              'RF OperatingIncome',  # 営業利益（単位：百万円）
              'RF OrdinaryIncome',  # 経常利益（単位：百万円）
              'RF NetIncome',  # 当期純利益（単位：百万円）
              'RF TotalAssets',  # 総資産（単位：百万円）
              'RF NetAssets',  # 純資産（単位：百万円）
              'RF CashFlowsFromOperating',  # 営業キャッシュフロー（単位：百万円）
              'RF CashFlowsFromFinancing',  # 財務キャッシュフロー（単位：百万円）
              'RF CashFlowsFromInvesting',  # 投資キャッシュフロー（単位：百万円）
              ]:
        g = stock_fin[latest].groupby('code')[c]
        for d in [1, 2, 3, 4, 5, 6, 7, 8]:
            stock_fin[c + f'_return{d}'] = np.nan
            r = g.pct_change(d)
            stock_fin.loc[stock_fin.index.isin(r.index), c + f'_return{d}'] = r

    # 過去数日にレポートが出されたかどうか
    timedelta = stock_fin.groupby('code')['datetime'].diff()
    # stock_fin['report_date_diff_in_7days'] = timedelta <= pd.Timedelta(7, 'D')
    # stock_fin['report_date_diff_in_14days'] = timedelta <= pd.Timedelta(14, 'D')
    # stock_fin['report_date_diff_in_21days'] = timedelta <= pd.Timedelta(21, 'D')
    stock_fin['report_date_diff_in_28days'] = timedelta <= pd.Timedelta(28, 'D')

    return stock_fin


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


class ScoringService(object):
    VAL_START = '2020-01-01'
    VAL_END = '2020-11-30'
    TEST_START = '2021-03-27'
    # 目的変数
    TARGET_LABELS = ['label_high_20', 'label_low_20']

    # モデルをこの変数に読み込む
    models = None
    nikkei225 = None
    normalize_value = None
    le_dict = None

    @classmethod
    def get_model(cls, model_path='../model'):
        if cls.models is None:
            cls.models = {}
        for label in cls.TARGET_LABELS:
            for i in range(3):
                model_name = f'final_lgb_{label}_seed{i}'
                model_file = os.path.join(model_path, model_name)
                cls.models[f'{label}_seed{i}'] = lgb.Booster(model_file=model_file)

        with open(os.path.join(model_path, 'nikkei225.txt'), 'r') as f:
            nikkei225 = f.readlines()
        cls.nikkei225 = sorted([int(s.strip()) for s in nikkei225])
        cls.normalize_value = pd.read_csv(os.path.join(model_path, 'normalize_value.csv'))

        with open(os.path.join(model_path, 'le_dict.pkl'), 'rb') as f:
            cls.le_dict = pickle.load(f)

        return True

    @classmethod
    def predict(cls, inputs, check_val_score=False):

        stock_list = pd.read_csv(inputs['stock_list'])
        stock_price = pd.read_csv(inputs['stock_price'])
        stock_price['datetime'] = pd.to_datetime(stock_price['EndOfDayQuote Date'])
        stock_price.set_index('datetime', inplace=True)
        stock_fin = pd.read_csv(inputs['stock_fin'])
        stock_fin['datetime'] = pd.to_datetime(stock_fin['base_date'])
        stock_fin.set_index('datetime', inplace=True)

        all_codes = stock_list.query('prediction_target')['Local Code'].unique()

        # stock_priceはstartのn日前からしか使わない
        margin = 240
        date_list = stock_price.index.unique()
        date_list = [s.strftime('%Y-%m-%d') for s in sorted(date_list[date_list < cls.VAL_START])]
        start_dt = date_list[-margin]
        print(start_dt)
        stock_price = stock_price[stock_price.index >= start_dt].copy()

        stock_fin_features = make_stock_fin_features(stock_fin, stock_list, stock_price, cls.normalize_value)
        stock_fin_features = reduce_mem_usage(stock_fin_features)
        stock_fin_features = stock_fin_features[stock_fin_features['datetime'] >= cls.VAL_START].copy()
        del stock_fin
        gc.collect()

        stock_price = reduce_mem_usage(stock_price)
        stock_price_features = make_stock_price_features(stock_price, stock_list, cls.nikkei225)
        stock_price_features = reduce_mem_usage(stock_price_features)
        del stock_price
        gc.collect()

        features = get_features(stock_price_features, stock_fin_features, all_codes)
        features.drop(drop_columns, axis=1, inplace=True)

        # categorical
        categorical_feature = [c for c in features.columns if features[c].dtype.name in ['object', 'category']]
        categorical_feature = [c for c in categorical_feature if not is_categorical(features[c])]
        for c in categorical_feature:
            features[c] = cls.le_dict[c].transform(fill_na_by_unique_value(features[c]))

        # debug用
        if check_val_score:
            print('Start checking val score')
            stock_labels = pd.read_csv(inputs['stock_labels'])
            stock_labels['datetime'] = pd.to_datetime(stock_labels['base_date'])
            stock_labels.set_index('datetime', inplace=True)

            X, y = get_dataframes(features, stock_labels)
            latest_idx = X['latest']
            X.drop(['code', 'latest'], axis=1, inplace=True)
            val_index = (X.index >= cls.VAL_START) & (X.index <= cls.VAL_END)
            val_index = val_index & latest_idx
            X, y = X[val_index], y[val_index]
            pred_dict = {}
            for label in cls.TARGET_LABELS:
                pred_dict[label] = np.mean([cls.models[f'{label}_seed{i}'].predict(X) for i in range(3)], axis=0)
            score, high_score, low_score = signate_metric(y, pred_dict)
            print('Score:', score)
            print('High spearmanr:', high_score)
            print('Low spearmanr:', low_score)
            return

        else:
            X = features.set_index('datetime')
            test_index = X.index >= cls.VAL_START
            X = X[test_index]
            # 結果を以下のcsv形式で出力する
            # １列目:datetimeとcodeをつなげたもの(Ex 2016-05-09-1301)
            # ２列目:label_high_20 終値→最高値への変化率
            # ３列目:label_low_20 終値→最安値への変化率
            # headerはなし、B列C列はfloat64

            # 日付と銘柄コードに絞り込み
            output_df = X[['code']].copy()
            # codeを出力形式の１列目と一致させる
            output_df['code'] = output_df.index.strftime('%Y-%m-%d-') + output_df['code'].astype(str)

            # 出力対象列を定義
            output_columns = ['code', 'label_high_20', 'label_low_20']

            X.drop(['code', 'latest'], axis=1, inplace=True)
            # 目的変数毎に予測
            for label in cls.TARGET_LABELS:
                output_df[label] = np.mean([cls.models[f'{label}_seed{i}'].predict(X) for i in range(3)], axis=0)

            out = io.StringIO()
            output_df.to_csv(out, header=False, index=False, columns=output_columns)

            return out.getvalue()
