import os

HOME_DIR = os.path.expanduser('~')
DATA_DIR = os.path.join(HOME_DIR, 'private/CS760/Project/Data')
AUDIO_DIR = os.path.join(DATA_DIR, 'fma_small')
MEL_SPEC_DIR = os.path.join(DATA_DIR, 'mel_spec')

N_MELS = 128