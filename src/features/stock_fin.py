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

    # ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????OK
    version, latest = get_version_on_stock_fin(stock_fin, True)
    stock_fin['version'] = version
    stock_fin['latest'] = latest
    stock_fin['first_repot'] = stock_fin['version'] == 1

    # ??????????????????nan???????????????????????????????????????
    tmp = stock_fin.groupby('code')['RD AnnualDividendPerShare'].mean()
    idx = stock_fin['code'].isin(tmp[tmp.isna()].index.values)
    stock_fin.loc[idx, 'RD AnnualDividendPerShare'] = stock_fin.loc[idx, 'RD QuarterlyDividendPerShare']
    # ?????????????????????nan??????0??????
    tmp = stock_fin.groupby('code')['RD AnnualDividendPerShare'].mean()
    idx = stock_fin['code'].isin(tmp[tmp.isna()].index.values)
    stock_fin.loc[idx, 'RD AnnualDividendPerShare'] = stock_fin.loc[idx, 'RD AnnualDividendPerShare'].fillna(0)
    # ????????????nan??????????????????????????????????????????????????????
    stock_fin['RD AnnualDividendPerShare'] = stock_fin.groupby('code')['RD AnnualDividendPerShare'].fillna(
        method='ffill')

    # ??????????????????nan???????????????????????????????????????
    tmp = stock_fin.groupby('code')['FD AnnualDividendPerShare'].mean()
    idx = stock_fin['code'].isin(tmp[tmp.isna()].index.values)
    stock_fin.loc[idx, 'FD AnnualDividendPerShare'] = stock_fin.loc[idx, 'FD QuarterlyDividendPerShare']
    # ?????????????????????nan??????0??????
    tmp = stock_fin.groupby('code')['FD AnnualDividendPerShare'].mean()
    idx = stock_fin['code'].isin(tmp[tmp.isna()].index.values)
    stock_fin.loc[idx, 'FD AnnualDividendPerShare'] = stock_fin.loc[idx, 'FD AnnualDividendPerShare'].fillna(0)
    # ????????????nan??????????????????????????????????????????????????????
    stock_fin['FD AnnualDividendPerShare'] = stock_fin.groupby('code')['FD AnnualDividendPerShare'].fillna(
        method='ffill')

    # ????????????
    stock_fin['market_cap'] = stock_fin['official close'] * stock_fin['IssuedShareEquityQuote IssuedShare']  # ????????????
    stock_fin['PER'] = stock_fin['official close'] / (stock_fin['RF NetIncome'] * 1_000_000 / stock_fin[
        'IssuedShareEquityQuote IssuedShare'])  # PER(???????????????)
    stock_fin.loc[stock_fin['RF CashFlowsFromOperating'] == 0, 'PER'] = np.nan  # ??????????????????????????????
    stock_fin['PBR'] = stock_fin['official close'] / (stock_fin['RF NetAssets'] * 1_000_000 / stock_fin[
        'IssuedShareEquityQuote IssuedShare'])  # PBR(?????????????????????)
    stock_fin['ROE'] = stock_fin['PBR'] / stock_fin['PER']  # ROE(?????????????????????)
    stock_fin['profit_margin'] = stock_fin['RF NetIncome'] / stock_fin['RF NetSales']  # ????????????
    stock_fin.loc[stock_fin['RF CashFlowsFromOperating'] == 0, 'profit_margin'] = np.nan
    stock_fin['equity_ratio'] = stock_fin['RF NetAssets'] / stock_fin['RF TotalAssets']  # ??????????????????
    stock_fin['dividend_yield'] = stock_fin['RD AnnualDividendPerShare'] / stock_fin[
        'official close'] * 100  # ???????????????
    stock_fin['sum_cache_flow'] = stock_fin['RF CashFlowsFromOperating'] + \
                                  stock_fin['RF CashFlowsFromFinancing'] + \
                                  stock_fin['RF CashFlowsFromInvesting']

    stock_fin['ForecastPER'] = stock_fin['official close'] / (stock_fin['FF NetIncome'] * 1_000_000 / stock_fin[
        'IssuedShareEquityQuote IssuedShare'])  # ??????PER
    stock_fin.loc[stock_fin['RF CashFlowsFromOperating'] == 0, 'ForecastPER'] = np.nan
    stock_fin['Forecastprofit_margin'] = stock_fin['FF NetIncome'] / stock_fin['FF NetSales']  # ??????????????????
    stock_fin.loc[stock_fin['RF CashFlowsFromOperating'] == 0, 'Forecastprofit_margin'] = np.nan
    stock_fin['Forecastdividend_yield'] = stock_fin['FD AnnualDividendPerShare'] / stock_fin[
        'official close'] * 100  # ?????????????????????

    # ????????????????????????21???????????????(=??????????????????20?????????????????????), n?????????????????????
    timedelta = pd.to_datetime(stock_fin['FD RecordDate']) - stock_fin['datetime']
    stock_fin['RecordDateIn21Days'] = timedelta <= pd.Timedelta(21, 'D')
    stock_fin['RecordDateIn41Days'] = timedelta <= pd.Timedelta(41, 'D')
    stock_fin['RecordDateIn61Days'] = timedelta <= pd.Timedelta(61, 'D')

    stock_fin.loc[
        stock_fin['FD RecordDate'].isna(), ['RecordDateIn21Days', 'RecordDateIn41Days', 'RecordDateIn61Days']] = np.nan

    # ????????????????????????????????????
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

    # ?????????
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

    # pct?????????
    for c in ['PER', 'PBR', 'ROE', 'profit_margin', 'equity_ratio', 'sum_cache_flow',
              'RF NetSales',  # ?????????????????????????????????
              'RF OperatingIncome',  # ????????????????????????????????????
              'RF OrdinaryIncome',  # ????????????????????????????????????
              'RF NetIncome',  # ???????????????????????????????????????
              'RF TotalAssets',  # ?????????????????????????????????
              'RF NetAssets',  # ?????????????????????????????????
              'RF CashFlowsFromOperating',  # ??????????????????????????????????????????????????????
              'RF CashFlowsFromFinancing',  # ??????????????????????????????????????????????????????
              'RF CashFlowsFromInvesting',  # ??????????????????????????????????????????????????????
              ]:
        g = stock_fin[latest].groupby('code')[c]
        for d in [1, 2, 3, 4, 5, 6, 7, 8]:
            stock_fin[c + f'_return{d}'] = np.nan
            r = g.pct_change(d)
            stock_fin.loc[stock_fin.index.isin(r.index), c + f'_return{d}'] = r

    # ??????????????????????????????????????????????????????
    timedelta = stock_fin.groupby('code')['datetime'].diff()
    # stock_fin['report_date_diff_in_7days'] = timedelta <= pd.Timedelta(7, 'D')
    # stock_fin['report_date_diff_in_14days'] = timedelta <= pd.Timedelta(14, 'D')
    # stock_fin['report_date_diff_in_21days'] = timedelta <= pd.Timedelta(21, 'D')
    stock_fin['report_date_diff_in_28days'] = timedelta <= pd.Timedelta(28, 'D')

    return stock_fin, normalize_value