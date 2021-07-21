import pickle
import warnings
from argparse import ArgumentParser
from pathlib import Path

import lightgbm as lgb
import numpy as np
from category_encoders.utils import convert_input, convert_input_vector
from nyaggle.ensemble import averaging
from nyaggle.experiment import Experiment, run_experiment
from nyaggle.experiment.auto_prep import autoprep_gbdt
from nyaggle.validation import TimeSeriesSplit
from pandas.api.types import is_categorical
from sklearn.preprocessing import LabelEncoder

from src.eval.metrics import signate_metric
from src.eval.spearmanr import custum_eval_sklearn
from src.eval.splits import get_ts1, get_ts2
from src.features.basic import get_all_codes, get_dataframes, get_features, get_inputs
from src.features.stock_fin import make_stock_fin_features
from src.features.stock_price import make_stock_price_features
from src.parameters.lgb import best_params_dict, fit_params, hpo_params, model_params
from src.utils.utils import fill_na_by_unique_value, reduce_mem_usage

warnings.simplefilter('ignore', UserWarning)

VAL_START = '2020-01-01'
VAL_END = '2020-11-30'
TARGET_LABELS = ['label_high_20', 'label_low_20']

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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('exp_name')
    parser.add_argument('--log_dir', default='./work_dirs')
    parser.add_argument('--data_dir', default='./data/jpx')
    parser.add_argument('--cache_dir', default='./work_dirs/features')
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--tune_hp', action='store_true')
    return parser.parse_args()


def main(args):
    log_dir = Path(args.log_dir) / args.exp_name
    if args.exp_name.startswith('exp'):
        experiment = Experiment(str(log_dir), if_exists='rename')
        log_dir = Path(experiment.logging_directory)
    else:
        experiment = Experiment(str(log_dir), if_exists='error')
        log_dir = Path(experiment.logging_directory)

    with experiment as exp:
        data_dir = Path(args.data_dir)

        # 日経225の銘柄コード一覧
        with open('./tools/infos/nikkei225.txt', 'r') as f:
            nikkei225 = f.readlines()
        nikkei225 = sorted([int(s.strip()) for s in nikkei225])

        dfs = get_inputs(data_dir)
        stock_list = dfs['stock_list']
        stock_price = dfs['stock_price']
        stock_fin = dfs['stock_fin']
        stock_labels = dfs['stock_labels']
        all_codes = get_all_codes(stock_list)

        stock_fin_features, normalize_value = make_stock_fin_features(stock_fin, stock_list, stock_price, None)
        normalize_value.to_csv(log_dir / 'normalize_value.csv', index=False)
        stock_fin_features = reduce_mem_usage(stock_fin_features)
        del stock_fin

        stock_price = reduce_mem_usage(stock_price)
        stock_price_features = make_stock_price_features(stock_price, stock_list, nikkei225)
        stock_price_features = reduce_mem_usage(stock_price_features)
        del stock_price

        # features = get_features(stock_price_features, stock_fin_features)
        features = get_features(stock_price_features, stock_fin_features, all_codes)  # codeしぼったほうが精度良さそう

        X, y = get_dataframes(features, stock_labels)

        # test用のindex
        code_idx = X['code'].isin(all_codes).values
        latest_idx = X['latest'].values
        target_idx = code_idx & latest_idx

        # columnsの処理
        X.drop(drop_columns, axis=1, inplace=True)
        X.drop(['code', 'latest'], axis=1, inplace=True)

        # trainのindexはTimeSeriesSplitで管理する
        X_train, y_train = X.copy(), y.copy()
        # testのindexはとりあえずpublic LBに合わせる
        test_mask = (X.index >= VAL_START) & (X.index <= VAL_END)
        test_mask = test_mask & target_idx
        X_test, y_test = X[test_mask], y[test_mask]

        categorical_feature = [c for c in X_train.columns if X_train[c].dtype.name in ['object', 'category']]
        X_train, X_test, le_dict = autoprep_gbdt('lgbm', X_train, X_test, categorical_feature)
        with open(log_dir / 'le_dict.pkl', 'wb') as f:
            pickle.dump(le_dict, f)

        if args.train_all:
            # 提出版はts2を使う
            ts = get_ts2(X_train, target_idx)
        else:
            ts = get_ts1(X_train, target_idx)

        if args.tune_hp:
            params_dict = {}
            for target_label in TARGET_LABELS:
                # 時間節約のために最後のsplitだけ使う
                hpo_ts = TimeSeriesSplit(X_train.index.to_series())
                hpo_ts.times = ts.times[-1:]
                result = run_experiment(model_params, X_train=X_train, y=y_train[target_label], X_test=X_test,
                                        cv=hpo_ts, fit_params=fit_params, eval_func=custum_eval_sklearn,
                                        with_auto_hpo=True, hpo_params=hpo_params,
                                        logging_directory=str(log_dir / f'lgb_hpo_{target_label}'))
                params_dict[target_label] = result.models[0].get_params()
        else:
            params_dict = best_params_dict

        exp.log('MODEL PARAMS:')
        for k, v in params_dict.items():
            exp.log(f'{k}:')
            exp.log(v)

        result_dict = {}
        pred_dict = {}

        for target_label in TARGET_LABELS:
            results = []
            for i in range(3):
                best_params = params_dict[target_label].copy()
                best_params['seed'] = i
                result = run_experiment(best_params, X_train=X_train, y=y_train[target_label], X_test=X_test,
                                        cv=ts, fit_params=fit_params, eval_func=custum_eval_sklearn,
                                        logging_directory=str(log_dir / f'lgb_{target_label}_seed{i}'))
                results.append(result)

            ensemble = averaging([result.test_prediction for result in results])
            pred_dict[target_label] = ensemble.test_prediction
            result_dict[target_label] = results

        if not args.train_all:
            metrics = signate_metric(y_test, pred_dict)
            exp.log(f'Score: {metrics[0]:.4f} spearmanr_high: {metrics[1]:.4f} spearmanr_low: {metrics[2]:.4f}')

        exp.log('Best iteration:')
        for target_label, results in result_dict.items():
            for i, result in enumerate(results):
                exp.log(f'{target_label}_{i}:')
                exp.log(', '.join([str(model.best_iteration_) for model in result.models]))
            mean_num_iterations = int(np.mean([result.models[-1].best_iteration_ for result in results]))
            exp.log(f'mean_num_iteration of {target_label}: {mean_num_iterations}')
            params_dict[target_label]['num_iterations'] = mean_num_iterations

        #
        # 全データを使ったtrain
        #
        if args.train_all:
            X_train, y_train = X, y
        else:
            # Public LBをターゲットに
            train_index = X.index < VAL_START
            X_train, y_train = X[train_index].copy(), y[train_index].copy()

        if 'early_stopping_rounds' in fit_params:
            del fit_params['early_stopping_rounds']
        for target_label in TARGET_LABELS:
            if 'early_stopping_round' in params_dict[target_label]:
                del params_dict[target_label]['early_stopping_round']

        # categorical
        categorical_feature = [c for c in X_train.columns if X_train[c].dtype.name in ['object', 'category']]
        categorical_feature = [c for c in categorical_feature if not is_categorical(X_train[c])]
        for c in categorical_feature:
            X_train[c] = LabelEncoder().fit_transform(fill_na_by_unique_value(X_train[c]))

        for target_label in TARGET_LABELS:
            train_x = convert_input(X_train)
            train_y = convert_input_vector(y_train[target_label], train_x.index)
            for i in range(3):
                best_params = params_dict[target_label].copy()
                best_params['seed'] = i
                estimator = lgb.LGBMRegressor(**best_params)
                estimator.fit(train_x, train_y, **fit_params)
                estimator.booster_.save_model(str(log_dir / f'final_lgb_{target_label}_seed{i}'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
