"""
Machine learning pipeline
"""

import pandas as pd

from sklearn import (
    pipeline,
    preprocessing,
    impute,
    compose,
    metrics,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import data_handler as dh

import time

from tqdm import tqdm

import random

import pickle

random.seed(dh.SEED)


def save_best_model(models: dict, ordered_results: pd.DataFrame, model_save_path: str) -> None:
    """Saves the best model in a pickle"""

    model_name = ordered_results['Model'][1]
    pickle.dump(models[model_name], open(model_save_path.format(model_name), 'wb'))


def train_and_eval(classifiers: dict,
                   train_file_path: str,
                   test_file_path: str,
                   model_save_path: str,
                   ):
    """Train a series of classifiers, evaluate them and save the best one"""

    train_data = pd.read_csv(train_file_path, sep='\t')
    train_target = train_data.pop('TARGET')

    test_data = pd.read_csv(test_file_path, sep='\t')
    test_target = test_data.pop('TARGET')

    results = pd.DataFrame({'Model': [], 'AUC': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})

    for model_name, model in tqdm(classifiers.items()):

        start_time = time.time()
        model.fit(train_data, train_target)
        total_time = time.time() - start_time

        pred = model.predict(test_data)

        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    {
                        'Model': [model_name],
                        'AUC': [metrics.roc_auc_score(test_target, pred) * 100],
                        'Accuracy': [metrics.accuracy_score(test_target, pred) * 100],
                        'Bal Acc.': [metrics.balanced_accuracy_score(test_target, pred) * 100],
                        'Time': [total_time]
                    }
                )
            ],
            ignore_index=True,
        )

    results_ord = results.sort_values(by=['AUC'], ascending=False, ignore_index=True)
    results_ord.index += 1
    results_ord.style.bar(subset=['AUC', 'Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')
    print(results_ord)

    save_best_model(classifiers, results_ord, model_save_path)


if __name__ == '__main__':

    dh.BOOLEAN_FIELDS.remove('ICU')

    BEST_MODEL_SAVE_PATH = './data/models/best_model_{}.pkl'

    NUMERICAL_PIPELINE = pipeline.Pipeline(
        steps=[
            ('imputer', impute.KNNImputer(missing_values=-1, n_neighbors=7)),
            ('scaler', preprocessing.RobustScaler()),
            ('normalizer', preprocessing.PowerTransformer()),
        ]
    )

    CATEGORICAL_BOOLEAN_PIPELINE = pipeline.Pipeline(
        steps=[
            ('imputer', impute.KNNImputer(missing_values=-1, n_neighbors=7)),
        ]
    )

    PREPROCESSING_PIPELINE = compose.ColumnTransformer(
        transformers=[
            ('numerical', NUMERICAL_PIPELINE, dh.NUMERICAL_FIELDS),
            ('categorical',
             CATEGORICAL_BOOLEAN_PIPELINE,
             dh.BOOLEAN_FIELDS + dh.CATEGORICAL_FIELDS
             ),
        ],
        remainder='drop',
    )

    TREE_CLASSIFIERS = {
        'Decision Tree': DecisionTreeClassifier(),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'AdaBoost': AdaBoostClassifier(n_estimators=100),
        'Skl GBM': GradientBoostingClassifier(n_estimators=100),
        'Skl HistGBM': HistGradientBoostingClassifier(max_iter=100),
        'XGBoost': XGBClassifier(n_estimators=100),
        'LightGBM': LGBMClassifier(n_estimators=100),
        'CatBoost': CatBoostClassifier(n_estimators=100, allow_writing_files=False, verbose=False),
    }
    TREE_CLASSIFIERS = {
        name: pipeline.make_pipeline(PREPROCESSING_PIPELINE, model)
        for name, model in TREE_CLASSIFIERS.items()
    }

    train_and_eval(TREE_CLASSIFIERS,
                   dh.TRAIN_FILE_PATH,
                   dh.TEST_FILE_PATH,
                   BEST_MODEL_SAVE_PATH,
                   )
