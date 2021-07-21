from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_importance(importance: pd.DataFrame, path: Optional[str] = None, top_n: int = 100,
                    figsize: Optional[Tuple[int, int]] = None,
                    title: Optional[str] = None):
    """
    Plot feature importance and write to image

    Args:
        importance:
            The dataframe which has "feature" and "importance" column
        path:
            The file path to be saved
        top_n:
            The number of features to be visualized
        figsize:
            The size of the figure
        title:
            The title of the plot
    Example:
        >>> import pandas as pd
        >>> import lightgbm as lgb
        >>> from nyaggle.util import plot_importance
        >>> from sklearn.datasets import make_classification

        >>> X, y = make_classification()
        >>> X = pd.DataFrame(X, columns=['col{}'.format(i) for i in range(X.shape[1])])
        >>> booster = lgb.train({'objective': 'binary'}, lgb.Dataset(X, y))
        >>> importance = pd.DataFrame({
        >>>     'feature': X.columns,
        >>>     'importance': booster.feature_importance('gain')
        >>> })
        >>> plot_importance(importance, 'importance.png')
    """
    importance = importance.groupby('feature')['importance'] \
        .mean() \
        .reset_index() \
        .sort_values(by='importance', ascending=False)

    if len(importance) > top_n:
        importance = importance.iloc[:top_n, :]

    if figsize is None:
        figsize = (10, 16)

    if title is None:
        title = 'Feature Importance'

    plt.figure(figsize=figsize)
    sns.barplot(x="importance", y="feature", data=importance)
    plt.title(title)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
