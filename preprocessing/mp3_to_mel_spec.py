import os

import numpy as np
import librosa

import config

def convert_all_audio_in_dir(subdir_path):
    for audio_file in os.listdir(subdir_path):
        audio_file_path = os.path.join(subdir_path, audio_file)

        y, sr = librosa.load(audio_file_path)
        spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_mels=config.N_MELS)
        log_spectrogram = librosa.logamplitude(spectrogram, ref_power=np.max)

        print(log_spectrogram)

if __name__ == '__main__':
    for subdir in os.listdir(config.AUDIO_DIR):
        subdir_path = os.path.join(config.AUDIO_DIR, subdir)
        if os.path.isdir(subdir_path):
            convert_all_audio_in_dir(subdir_path)
