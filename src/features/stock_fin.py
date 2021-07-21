import numpy as np
import pandas as pd

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
    normalize_value = stock_fin[stock_fin.datetime <= '2021-03-24'].groupby(['33 Sector(Code)'])[normalize_target].mean().reset_index()
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

    return stock_fin, normalize_value