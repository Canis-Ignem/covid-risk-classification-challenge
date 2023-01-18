"""
Applies the data processing and saves it for the DL model
"""

import pandas as pd

from sklearn import (
    pipeline,
    preprocessing,
    impute,
    compose,
)

import data_handler as dh


if __name__ == '__main__':

    dh.BOOLEAN_FIELDS.remove('ICU')

    PROCESSED_TRAIN_DATA = './data/train_data/processed_train_data.tsv'
    PROCESSED_TEST_DATA = './data/test_data/processed_test_data.tsv'

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
            ('numerical', NUMERICAL_PIPELINE, ['AGE']),
            ('categorical',
             CATEGORICAL_BOOLEAN_PIPELINE,
             ['PREGNANT', 'INTUBED'],
             ),
        ],
        remainder='passthrough',
    )

    train_data = pd.read_csv(dh.TRAIN_FILE_PATH, sep='\t')
    train_target = train_data.pop('TARGET')
    headers = train_data.columns

    test_data = pd.read_csv(dh.TEST_FILE_PATH, sep='\t')
    test_target = test_data.pop('TARGET')

    train_data = PREPROCESSING_PIPELINE.fit_transform(train_data)
    test_data = PREPROCESSING_PIPELINE.transform(test_data)

    train_data = pd.DataFrame(train_data, columns=headers)
    train_data['TARGET'] = train_target

    test_data = pd.DataFrame(test_data, columns=headers)
    test_data['TARGET'] = test_target

    train_data.to_csv(PROCESSED_TRAIN_DATA, sep='\t', index=False)
    test_data.to_csv(PROCESSED_TEST_DATA, sep='\t', index=False)
