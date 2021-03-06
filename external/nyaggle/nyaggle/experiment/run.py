import os
import pickle
import time
from collections import namedtuple
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union

import pandas as pd
import sklearn.utils.multiclass as multiclass
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss
from sklearn.model_selection import BaseCrossValidator

from nyaggle.environment import requires_catboost, requires_lightgbm, requires_xgboost
from nyaggle.experiment.auto_prep import autoprep_gbdt
from nyaggle.experiment.experiment import Experiment
from nyaggle.experiment.hyperparameter_tuner import find_best_lgbm_parameter
from nyaggle.feature_store import load_features
from nyaggle.util import plot_importance, is_gbdt_instance, make_submission_df
from nyaggle.validation.cross_validate import cross_validate
from nyaggle.validation.split import check_cv

ExperimentResult = namedtuple('ExperimentResult',
                              [
                                  'oof_prediction',
                                  'test_prediction',
                                  'metrics',
                                  'models',
                                  'importance',
                                  'time',
                                  'submission_df'
                              ])


class ExpeimentProxy(object):
    __slots__ = ["_obj", "__weakref__"]

    def __init__(self, obj):
        object.__setattr__(self, "_obj", obj)

    def __getattribute__(self, name):
        return getattr(object.__getattribute__(self, "_obj"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_obj"), name, value)

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, trace):
        pass


def run_experiment(model_params: Dict[str, Any],
                   X_train: pd.DataFrame, y: pd.Series,
                   X_test: Optional[pd.DataFrame] = None,
                   logging_directory: str = 'output/{time}',
                   if_exists: str = 'error',
                   eval_func: Optional[Callable] = None,
                   algorithm_type: Union[str, Type[BaseEstimator]] = 'lgbm',
                   fit_params: Optional[Union[Dict[str, Any], Callable]] = None,
                   cv: Optional[Union[int, Iterable, BaseCrossValidator]] = None,
                   groups: Optional[pd.Series] = None,
                   categorical_feature: Optional[List[str]] = None,
                   sample_submission: Optional[pd.DataFrame] = None,
                   submission_filename: Optional[str] = None,
                   type_of_target: str = 'auto',
                   feature_list: Optional[List[Union[int, str]]] = None,
                   feature_directory: Optional[str] = None,
                   inherit_experiment: Optional[Experiment] = None,
                   with_auto_hpo: bool = False,
                   with_auto_prep: bool = False,
                   with_mlflow: bool = False,
                   hpo_params: dict = None,
                   ):
    """
    Evaluate metrics by cross-validation and stores result
    (log, oof prediction, test prediction, feature importance plot and submission file)
    under the directory specified.

    One of the following estimators are used (automatically dispatched by ``type_of_target(y)`` and ``gbdt_type``).

    * LGBMClassifier
    * LGBMRegressor
    * CatBoostClassifier
    * CatBoostRegressor

    The output files are laid out as follows:

    .. code-block:: none

      <logging_directory>/
          log.txt                  <== Logging file
          importance.png           <== Feature importance plot generated by nyaggle.util.plot_importance
          oof_prediction.npy       <== Out of fold prediction in numpy array format
          test_prediction.npy      <== Test prediction in numpy array format
          submission.csv           <== Submission csv file
          metrics.json             <== Metrics
          params.json              <== Parameters
          models/
              fold1                <== The trained model in fold 1
              ...

    Args:
        model_params:
            Parameters passed to the constructor of the classifier/regressor object (i.e. LGBMRegressor).
        X_train:
            Training data. Categorical feature should be casted to pandas categorical type or encoded to integer.
        y:
            Target
        X_test:
            Test data (Optional). If specified, prediction on the test data is performed using ensemble of models.
        logging_directory:
            Path to directory where output of experiment is stored.
        if_exists:
            How to behave if the logging directory already exists.

            - error: Raise a ValueError.
            - replace: Delete logging directory before logging.
            - append: Append to exisitng experiment.
            - rename: Rename current directory by adding "_1", "_2"... prefix
        fit_params:
            Parameters passed to the fit method of the estimator. If dict is passed, the same parameter except
            eval_set passed for each fold. If callable is passed,
            returning value of ``fit_params(fold_id, train_index, test_index)`` will be used for each fold.
        eval_func:
            Function used for logging and calculation of returning scores.
            This parameter isn't passed to GBDT, so you should set objective and eval_metric separately if needed.
            If ``eval_func`` is None, ``roc_auc_score`` or ``mean_squared_error`` is used by default.
        gbdt_type:
            Type of gradient boosting library used. "lgbm" (lightgbm) or "cat" (catboost)
        cv:
            int, cross-validation generator or an iterable which determines the cross-validation splitting strategy.

            - None, to use the default ``KFold(5, random_state=0, shuffle=True)``,
            - integer, to specify the number of folds in a ``(Stratified)KFold``,
            - CV splitter (the instance of ``BaseCrossValidator``),
            - An iterable yielding (train, test) splits as arrays of indices.
        groups:
            Group labels for the samples. Only used in conjunction with a ???Group??? cv instance (e.g., ``GroupKFold``).
        sample_submission:
            A sample dataframe alined with test data (Usually in Kaggle, it is available as sample_submission.csv).
            The submission file will be created with the same schema as this dataframe.
        submission_filename:
            The name of submission file will be created under logging directory. If ``None``, the basename of the logging
            directory will be used as a filename.
        categorical_feature:
            List of categorical column names. If ``None``, categorical columns are automatically determined by dtype.
        type_of_target:
            The type of target variable. If ``auto``, type is inferred by ``sklearn.utils.multiclass.type_of_target``.
            Otherwise, ``binary``, ``continuous``, or ``multiclass`` are supported.
        feature_list:
            The list of feature ids saved through nyaggle.feature_store module.
        feature_directory:
            The location of features stored. Only used if feature_list is not empty.
        inherit_experiment:
            An experiment object which is used to log results. if not ``None``, all logs in this function are treated
            as a part of this experiment.
        with_auto_prep:
            If True, the input datasets will be copied and automatic preprocessing will be performed on them.
            For example, if ``gbdt_type = 'cat'``, all missing values in categorical features will be filled.
        with_auto_hpo:
            If True, model parameters will be automatically updated using optuna (only available in lightgbm).
        with_mlflow:
            If True, `mlflow tracking <https://www.mlflow.org/docs/latest/tracking.html>`_ is used.
            One instance of ``nyaggle.experiment.Experiment`` corresponds to one run in mlflow.
            Note that all output
            mlflow's directory (``mlruns`` by default).
    :return:
        Namedtuple with following members

        * oof_prediction:
            numpy array, shape (len(X_train),) Predicted value on Out-of-Fold validation data.
        * test_prediction:
            numpy array, shape (len(X_test),) Predicted value on test data. ``None`` if X_test is ``None``
        * metrics:
            list of float, shape(nfolds+1) ``metrics[i]`` denotes validation score in i-th fold.
            ``metrics[-1]`` is overall score.
        * models:
            list of objects, shape(nfolds) Trained models for each folds.
        * importance:
            list of pd.DataFrame, feature importance for each fold (type="gain").
        * time:
            Training time in seconds.
        * submit_df:
            The dataframe saved as submission.csv
    """
    start_time = time.time()
    cv = check_cv(cv, y)

    if feature_list:
        X = pd.concat([X_train, X_test]) if X_test is not None else X_train
        X.reset_index(drop=True, inplace=True)
        X = load_features(X, feature_list, directory=feature_directory)
        ntrain = len(X_train)
        X_train, X_test = X.iloc[:ntrain, :], X.iloc[ntrain:, :].reset_index(drop=True)

    _check_input(X_train, y, X_test)

    if categorical_feature is None:
        categorical_feature = [c for c in X_train.columns if X_train[c].dtype.name in ['object', 'category']]

    if type_of_target == 'auto':
        type_of_target = multiclass.type_of_target(y)
    model_type, eval_func, cat_param_name = _dispatch_models(algorithm_type, type_of_target, eval_func)

    if with_auto_prep:
        assert algorithm_type in ('cat', 'xgb', 'lgbm'), "with_auto_prep is only supported for gbdt"
        X_train, X_test, le_dict = autoprep_gbdt(algorithm_type, X_train, X_test, categorical_feature)

    logging_directory = logging_directory.format(time=datetime.now().strftime('%Y%m%d_%H%M%S'))

    if inherit_experiment is not None:
        experiment = ExpeimentProxy(inherit_experiment)
    else:
        experiment = Experiment(logging_directory, if_exists=if_exists, with_mlflow=with_mlflow)

    with experiment as exp:
        exp.log('Algorithm: {}'.format(algorithm_type))
        exp.log('Experiment: {}'.format(exp.logging_directory))
        exp.log('Params: {}'.format(model_params))
        exp.log('Features: {}'.format(list(X_train.columns)))
        exp.log_param('algorithm_type', algorithm_type)
        exp.log_param('num_features', X_train.shape[1])
        if callable(fit_params):
            exp.log_param('fit_params', str(fit_params))
        else:
            exp.log_dict('fit_params', fit_params)
        exp.log_dict('model_params', model_params)
        if feature_list is not None:
            exp.log_param('features', feature_list)

        if with_auto_hpo:
            assert algorithm_type == 'lgbm', 'auto-tuning is only supported for LightGBM'
            model_params = find_best_lgbm_parameter(model_params, X_train, y, cv=cv, groups=groups,
                                                    type_of_target=type_of_target, train_params=hpo_params)
            exp.log_param('model_params_tuned', model_params)

        exp.log('Categorical: {}'.format(categorical_feature))

        models = [model_type(**model_params) for _ in range(cv.get_n_splits())]

        if fit_params is None:
            fit_params = {}
        if cat_param_name is not None and not callable(fit_params) and cat_param_name not in fit_params:
            fit_params[cat_param_name] = categorical_feature

        if isinstance(fit_params, Dict):
            exp.log_params(fit_params)

        result = cross_validate(models, X_train=X_train, y=y, X_test=X_test, cv=cv, groups=groups,
                                logger=exp.get_logger(), eval_func=eval_func, fit_params=fit_params,
                                type_of_target=type_of_target)

        # save oof
        exp.log_numpy('oof_prediction', result.oof_prediction)
        exp.log_numpy('test_prediction', result.test_prediction)

        for i in range(cv.get_n_splits()):
            exp.log_metric('Fold {}'.format(i + 1), result.scores[i])
        exp.log_metric('Overall', result.scores[-1])

        # save importance plot
        if result.importance:
            importance = pd.concat(result.importance)
            plot_file_path = os.path.join(exp.logging_directory, 'importance.png')
            plot_importance(importance, plot_file_path)
            exp.log_artifact(plot_file_path)

        # save trained model
        for i, model in enumerate(models):
            _save_model(model, exp.logging_directory, i + 1, exp)

        # save submission.csv
        submit_df = None
        if X_test is not None:
            submit_df = make_submission_df(result.test_prediction, sample_submission, y)
            exp.log_dataframe(submission_filename or os.path.basename(exp.logging_directory), submit_df, 'csv')

        elapsed_time = time.time() - start_time

        return ExperimentResult(result.oof_prediction, result.test_prediction,
                                result.scores, models, result.importance, elapsed_time, submit_df)


def _dispatch_eval_func(target_type: str, custom_eval: Optional[Callable] = None):
    default_eval_func = {
        'binary': roc_auc_score,
        'multiclass': log_loss,
        'continuous': mean_squared_error
    }
    return custom_eval if custom_eval is not None else default_eval_func[target_type]


def _dispatch_gbdt_class(algorithm_type: str, type_of_target: str):
    is_regression = type_of_target == 'continuous'

    if algorithm_type == 'lgbm':
        requires_lightgbm()
        from lightgbm import LGBMClassifier, LGBMRegressor
        return LGBMRegressor if is_regression else LGBMClassifier
    elif algorithm_type == 'cat':
        requires_catboost()
        from catboost import CatBoostClassifier, CatBoostRegressor
        return CatBoostRegressor if is_regression else CatBoostClassifier
    else:
        requires_xgboost()
        assert algorithm_type == 'xgb'
        from xgboost import XGBClassifier, XGBRegressor
        return XGBRegressor if is_regression else XGBClassifier


def _dispatch_models(algorithm_type: Union[str, Type[BaseEstimator]],
                     target_type: str, custom_eval: Optional[Callable] = None):
    if not isinstance(algorithm_type, str):
        assert issubclass(algorithm_type, BaseEstimator), "algorithm_type should be str or subclass of BaseEstimator"
        return algorithm_type, _dispatch_eval_func(target_type, custom_eval), None

    cat_features = {
        'lgbm': 'categorical_feature',
        'cat': 'cat_features',
        'xgb': None
    }

    gbdt_class = _dispatch_gbdt_class(algorithm_type, target_type)
    eval_func = _dispatch_eval_func(target_type, custom_eval)

    return gbdt_class, eval_func, cat_features[algorithm_type]


def _save_model(model: BaseEstimator, logging_directory: str, fold: int, exp: Experiment):
    model_dir = os.path.join(logging_directory, 'models')
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, 'fold{}'.format(fold))

    if is_gbdt_instance(model, 'lgbm'):
        model.booster_.save_model(path)
    elif is_gbdt_instance(model, ('xgb', 'cat')):
        model.save_model(path)
    else:
        with open(path, "wb") as f:
            pickle.dump(model, f)

    exp.log_artifact(path)


def _check_input(X_train: pd.DataFrame, y: pd.Series,
                 X_test: Optional[pd.DataFrame] = None):
    assert len(X_train) == len(y), "length of X_train and y are different. len(X_train) = {}, len(y) = {}".format(
        len(X_train), len(y)
    )

    if X_test is not None:
        assert list(X_train.columns) == list(X_test.columns), "columns are different between X_train and X_test"
