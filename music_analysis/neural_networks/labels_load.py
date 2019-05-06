import pandas as pd

import config
import os

def essentia_file_to_labels():
    '''
        Function I made to convert a full essentia file to just
        A CSV w/ tracks & labels. Run this guy before training NNs,
        so we only have to store essential features in memory
    '''
    csv_path = config.FEATURES_WITH_LABELS
    data = pd.read_csv(csv_path)
    data_labels = data.loc[:, config.DF_COLUMNS]
    data_labels.to_csv(config.LABELS_ONLY, index=False)

def all_essentia_files_to_labels():
    # csv_paths = config.TRAIN_FOLDS
    # # csv_paths = config.TEST_FOLDS
    # for i, csv_path in enumerate(csv_paths):
    #     data = pd.read_csv(csv_path)
    #     data_labels = data.loc[:, config.DF_COLUMNS]
    #     data_labels.to_csv(config.TRAIN_FOLDS_LABELS[i], index = False)

    train_path = config.FULL_TRAIN
    data = pd.read_csv(train_path)
    data_labels = data.loc[:, config.DF_COLUMNS]
    data.to_csv(config.FULL_TRAIN_LABELS, index = False)

    test_path = config.FULL_TEST
    data = pd.read_csv(test_path)
    data_labels = data.loc[:, config.DF_COLUMNS]
    data.to_csv(config.FULL_TEST_LABELS, index = False)

def load_labels_cfg():
    '''
        Test FN - load labels from the config file
    '''
    label_path = config.LABELS_ONLY
    data = pd.read_csv(label_path)
    return data

def load_labels(filepath):
    '''
        Load the labels at a given CSV filepath & return
        them
    '''
    if not os.path.exists(filepath):
        raise Exception("CSV file <" + filepath + "> doesn't exist!")

    return pd.read_csv(filepath)

if __name__ == '__main__':
    # if not os.path.exists(config.LABELS_ONLY):
    #     essentia_file_to_labels()
    # data = load_labels(config.FEATURES_WITH_LABELS)
    # print(data.head(1000).loc[:, 'Year'])
    # all_essentia_files_to_labels()
    all_essentia_files_to_labels()
