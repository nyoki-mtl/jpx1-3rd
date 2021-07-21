from src.eval.spearmanr import custum_eval, custum_eval_sklearn

model_params = {
    'objective': 'regression',
    'metric': 'spearmanr',
    # 'learning_rate': 0.01,
}

# sklearn interfaceでfitされるときに渡すパラメタ
fit_params = {
    'eval_metric': custum_eval_sklearn,
    'early_stopping_rounds': 100
}

# optunaを通してlgb.trainに渡されるパラメタ
hpo_params = {
    'feval': custum_eval
}

best_params1 = {
    'objective': 'regression',
    'metric': 'spearmanr',
    'feature_pre_filter': False,
    'verbosity': -1,
    'lambda_l1': 0.16889530286745985,
    'lambda_l2': 1.2410595580996777e-06,
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.9000256584722999,
    'bagging_freq': 5,
    'min_child_samples': 5,
    'num_iterations': 10000,
}

best_params2 = {
    'objective': 'regression',
    'metric': 'spearmanr',
    'feature_pre_filter': False,
    'verbosity': -1,
    'lambda_l1': 8.062872033881925,
    'lambda_l2': 0.026625655716661895,
    'num_leaves': 92,
    'feature_fraction': 0.5,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'min_child_samples': 25,
    'num_iterations': 10000,
}

best_params_dict = {
    'label_high_20':
        {
            # 'boosting_type': 'gbdt',
            # 'class_weight': None,
            # 'colsample_bytree': 1.0,
            # 'importance_type': 'split',
            # 'learning_rate': 0.1,
            # 'max_depth': -1,
            # 'min_child_samples': 20,
            # 'min_child_weight': 0.001,
            # 'min_split_gain': 0.0,
            # 'n_estimators': 100,
            # 'n_jobs': -1,
            'objective': 'regression',
            'metric': 'spearmanr',
            # 'random_state': None,
            # 'reg_alpha': 0.0,
            # 'reg_lambda': 0.0,
            # 'silent': True,
            # 'subsample': 1.0,
            # 'subsample_for_bin': 200000,
            # 'subsample_freq': 0,
            'feature_pre_filter': False,
            'verbosity': -1,
            'lambda_l1': 0.0035375367055248907,
            'lambda_l2': 0.24776283117729828,
            'num_leaves': 253,
            'feature_fraction': 0.5,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'num_iterations': 10000,
            # 'early_stopping_round': 100
        },
    'label_low_20':
        {
            # 'boosting_type': 'gbdt',
            # 'class_weight': None,
            # 'colsample_bytree': 1.0,
            # 'importance_type': 'split',
            # 'learning_rate': 0.1,
            # 'max_depth': -1,
            # 'min_child_samples': 20,
            # 'min_child_weight': 0.001,
            # 'min_split_gain': 0.0,
            # 'n_estimators': 100,
            # 'n_jobs': -1,
            'objective': 'regression',
            'metric': 'spearmanr',
            # 'random_state': None,
            # 'reg_alpha': 0.0,
            # 'reg_lambda': 0.0,
            # 'silent': True,
            # 'subsample': 1.0,
            # 'subsample_for_bin': 200000,
            # 'subsample_freq': 0,
            'feature_pre_filter': False,
            'verbosity': -1,
            'lambda_l1': 0.0005732579932630379,
            'lambda_l2': 4.614903511074905e-06,
            'num_leaves': 31,
            'feature_fraction': 0.6,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'num_iterations': 10000,
            # 'early_stopping_round': 100
        }
}
