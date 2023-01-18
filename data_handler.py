"""
Proccess and handle data
"""

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import random


# Ensure consistency
SEED = 87324554
random.seed(SEED)

# Relevant files
TRAIN_FILE_PATH = './data/train_data/train_data.tsv'
TEST_FILE_PATH = './data/test_data/test_data.tsv'
DISCARDED_PATH = './data/discarded/discarded_data.tsv'

# Variable types
NUMERICAL_FIELDS = ['AGE']
DATE_FIELDS = ['DATE_DIED']
CATEGORICAL_FIELDS = ['CLASIFFICATION_FINAL', 'MEDICAL_UNIT']
BOOLEAN_FIELDS = ['USMER',
                  'SEX',
                  'PATIENT_TYPE',
                  'INTUBED',
                  'PNEUMONIA',
                  'PREGNANT',
                  'DIABETES',
                  'COPD',
                  'ASTHMA',
                  'INMSUPR',
                  'HIPERTENSION',
                  'OTHER_DISEASE',
                  'CARDIOVASCULAR',
                  'OBESITY',
                  'RENAL_CHRONIC',
                  'TOBACCO',
                  'ICU',
                  ]


def transform_boolean(x: int) -> int:
    """
    Convert boolean fields to ds friendly numbers
    1: 1
    2: 0
    else: -1
    """
    if x == 1:
        return 1
    elif x == 2:
        return 0
    return -1


def covid_classification_to_to_ordinal(x: int) -> int:
    """
    Transforms the covid classification into an ordinal
    4-90: 0 covid free
    1-3: 1-3 covid positive in increa
    else: -1 missing
    """
    if x >= 4 and x <= 90:
        return 0
    elif x > 90:
        return -1
    return x


def transform_date_died_to_boolean(x: str) -> int:
    """
    Transforms a date into boolean
    9999-99-99: 0 alive
    date: 1 dead
    else: -1 missing
    """

    if x == '9999-99-99':
        return 0
    elif '/' in x:
        return 1
    return -1


def create_target_column(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a target column based on the relevant columns
    0 -> no risk
    1 -> at risk
    """

    target = []

    for i in range(data.shape[0]):
        if data['DATE_DIED'][i] == 1 or data['ICU'][i] == 1:
            target.append(1)
        elif data['DATE_DIED'][i] != 1 and data['ICU'][i] == 0:
            target.append(0)
        elif data['DATE_DIED'][i] != 1 and data['ICU'][i] == -1:
            target.append(-1)

    data.drop(columns=['DATE_DIED', 'ICU'], inplace=True)
    data['TARGET'] = target

    return data


def input_male_pregnant_case(data: pd.DataFrame) -> pd.DataFrame:
    """Set missing male pregnancy cases to 0"""

    for i in range(data.shape[0]):
        if data['SEX'][i] == 0 and data['PREGNANT'][i] >= 90:
            data['PREGNANT'][i] = 0

    return data


def preprocess_data(input_file_path: str,
                    output_file_path: str = None
                    ) -> pd.DataFrame:
    """preprocess the into data into a more data science friendly format"""

    data = pd.read_csv(input_file_path, sep=',')

    data = input_male_pregnant_case(data)

    for field in BOOLEAN_FIELDS:
        data[field] = data[field].apply(lambda x: transform_boolean(x))

    for field in DATE_FIELDS:
        data[field] = data[field].apply(lambda x: transform_date_died_to_boolean(x))

    data = create_target_column(data)

    if output_file_path:
        data.to_csv(output_file_path, sep='\t', index=False)

    return data


def load_processed_data(input_file_path: str, discarded_file_path: str = None) -> pd.DataFrame:
    """Load processed data and discard rows with a missing target"""

    discarded_rows = []
    data = []

    with open(input_file_path, 'r', encoding='utf-8') as input_file:

        header = input_file.readline().strip().split('\t')

        for line in input_file:
            data_points = line.strip().split('\t')
            if data_points[-1] != '-1':
                data.append(data_points)
            else:
                discarded_rows.append(data_points)

        data = pd.DataFrame(data, columns=header)

    if discarded_rows:
        with open(discarded_file_path, 'w', encoding='utf-8') as discarded_file:
            discarded_file.write('\t'.join(header) + '\n')
            for row in discarded_rows:
                discarded_file.write('\t'.join(row) + '\n')

    return data


def create_train_test_split(input_file_path: str,
                            train_file_path: str,
                            test_file_path: str,
                            discarded_file_path: None,
                            test_size: int = 0.2,
                            ):
    """Split the data into train and test and discard rows with missing data"""

    data = load_processed_data(input_file_path=input_file_path,
                               discarded_file_path=discarded_file_path,
                               )

    target = data.pop('TARGET')

    x_train, x_test, y_train, y_test = train_test_split(data,
                                                        target,
                                                        test_size=test_size,
                                                        random_state=SEED,
                                                        stratify=target,
                                                        )

    x_train['TARGET'] = y_train
    x_train.to_csv(train_file_path,
                   sep='\t',
                   index=False,
                   )

    x_test['TARGET'] = y_test
    x_test.to_csv(test_file_path,
                  sep='\t',
                  index=False,
                  )


def load_train_test_data(train_file_path: str, test_file_path: str):
    """Load train and test files and returns batchified tensors"""

    train_data = []
    train_labels = []

    test_data = []
    test_labels = []
    with open(train_file_path, 'r', encoding='utf-8') as train_file:

        _ = train_file.readline()
        for line in train_file:
            data_points = line.strip().split('\t')
            train_data.append(data_points[:-1])
            train_labels.append(float(data_points[-1]))

    with open(test_file_path, 'r', encoding='utf-8') as test_file:

        _ = test_file.readline()
        for line in test_file:
            data_points = line.strip().split('\t')
            test_data.append(data_points[:-1])
            test_labels.append(float(data_points[-1]))

    train_data = np.array(train_data).astype(float)
    train_labels = np.array(train_labels)

    test_data = np.array(test_data).astype(float)
    test_labels = np.array(test_labels)

    return (
        train_data,
        train_labels,
        test_data,
        test_labels,
    )


if __name__ == '__main__':

    INPUT_PATH = './data/input/Data.csv'
    PROCESSED_PATH = './data/input/processed_data.tsv'

    preprocess_data(INPUT_PATH, PROCESSED_PATH)

    create_train_test_split(
        PROCESSED_PATH,
        TRAIN_FILE_PATH,
        TEST_FILE_PATH,
        DISCARDED_PATH,
    )
