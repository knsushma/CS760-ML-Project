# Change these depending on installation
import os

DATA_DIR = 'C:\\Users\\Matthew\\Desktop\\CS760\\Data'
MEL_SPEC_DIR = os.path.join(DATA_DIR, 'mel_spec')
FEATURES_WITH_LABELS = os.path.join(DATA_DIR, 'essentia_features_with_labels.csv')
LABELS_ONLY = os.path.join(DATA_DIR, 'track_labels.csv')

DF_COLUMNS = ['track_id', 'Genre', 'Artist', 'Year']

IMG_DIMS = [480, 640, 4]
IMG_DEPTH = 255