from scipy.stats import spearmanr


def custum_eval(y_pred, y_true):
    y_true = y_true.get_label()
    # こっちはうまくいかない... optuna/integration/_lightgbm_tuner/optimize.pyのL146 higher_is_betterが問題
    # custum funcは全部higher_is_better=Falseになる
    # metric = spearmanr(y_true, y_pred)[0]
    # return 'spearmanr', metric, True
    metric = - spearmanr(y_true, y_pred)[0]
    return 'spearmanr', metric, False


def custum_eval_sklearn(y_true, y_pred):
    # sklearn interface version
    metric = spearmanr(y_true, y_pred)[0]
    return 'spearmanr', metric, True
