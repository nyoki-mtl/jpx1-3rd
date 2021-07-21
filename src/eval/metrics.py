from scipy.stats import spearmanr


def signate_metric(y_true, y_pred):
    high_score = spearmanr(y_true['label_high_20'].values, y_pred['label_high_20'])[0]
    low_score = spearmanr(y_true['label_low_20'].values, y_pred['label_low_20'])[0]
    return (high_score - 1) ** 2 + (low_score - 1) ** 2, high_score, low_score
