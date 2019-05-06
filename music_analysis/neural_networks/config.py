# Change these depending on installation
import os

DATA_DIR = 'C:\\Users\\Matthew\\Desktop\\CS760\\Data'
MEL_SPEC_DIR = os.path.join(DATA_DIR, 'mel_spec')
FEATURES_WITH_LABELS = os.path.join(DATA_DIR, 'essentia_features_with_labels.csv')
LABELS_ONLY = os.path.join(DATA_DIR, 'track_labels.csv')

SPLITS_DIR = os.path.join(DATA_DIR, 'dataset_splits')
TRAIN_FOLDS = [os.path.join(SPLITS_DIR, 'essentia_trainfold_' + str(i) + '.csv') for i in range(1,6)]
TEST_FOLDS = [os.path.join(SPLITS_DIR, 'essentia_testfold_' + str(i) + '.csv') for i in range(1,6)]
TRAIN_FOLDS_LABELS = [os.path.join(SPLITS_DIR, 'essentia_trainfold_' + str(i) + '_labels.csv') for i in range(1,6)]
TEST_FOLDS_LABELS = [os.path.join(SPLITS_DIR, 'essentia_testfold_' + str(i) + '_labels.csv') for i in range(1,6)]

FULL_TRAIN = os.path.join(SPLITS_DIR, 'essentia_train.csv')
FULL_TRAIN_LABELS = os.path.join(SPLITS_DIR, 'essentia_train_labels.csv')
FULL_TEST = os.path.join(SPLITS_DIR, 'essentia_test.csv')
FULL_TEST_LABELS = os.path.join(SPLITS_DIR, 'essentia_test_labels.csv')

DF_COLUMNS = ['track_id', 'Top Genre', 'Artist', 'Year']

GENRES = ['Instrumental', 'Hip-Hop', 'Pop', 'Electronic', 'Experimental', 'Folk', 'Rock', 'International']

MIN_YEAR = 1980
MAX_YEAR = 2017

IMG_DIMS = [480, 640, 4]
IMG_DEPTH = 255

import pandas as pd

df = pd.read_csv(LABELS_ONLY)
ARTISTS = list(set(list(df.loc[:, 'Artist'])))
NUM_ARTISTS = len(ARTISTS)