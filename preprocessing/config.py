import os

HOME_DIR = os.path.expanduser('~')
DATA_DIR = os.path.join(HOME_DIR, 'private/CS760/Project/Data')
AUDIO_DIR = os.path.join(DATA_DIR, 'fma_small')
MEL_SPEC_DIR = os.path.join(DATA_DIR, 'mel_spec')
ESS_DIR = os.path.join(DATA_DIR, 'essentia_features')
N_MELS = 128
