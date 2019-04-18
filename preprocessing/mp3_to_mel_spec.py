import os

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import config
import pylab

def convert_all_audio_in_dir(subdir_path):
    
    dir_name = os.path.basename(os.path.normpath(subdir_path))
    
    for audio_file in os.listdir(subdir_path):
        audio_file_path = os.path.join(subdir_path, audio_file)

        y, sr = librosa.load(audio_file_path)
        
        #pylab.axes('off') #no axis
        pylab.axes([0.,0.,1.,1.], frameon=False, xticks=[], yticks=[]) #remove white-edge
        
        spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_mels=config.N_MELS)
        
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        librosa.display.specshow(log_spectrogram)

        #saving melspectogram as image
        mel_dir_path = os.path.join(config.MEL_SPEC_DIR, dir_name)
        if not os.path.exists(mel_dir_path):
            os.makedirs(mel_dir_path)

        mel_file_path = os.path.join(mel_dir_path, audio_file[:-3]+"png")
        pylab.savefig(mel_file_path, bbox_inches=None, pad_inches=0)
        pylab.close()
        

if __name__ == '__main__':
    for subdir in os.listdir(config.AUDIO_DIR):
        subdir_path = os.path.join(config.AUDIO_DIR, subdir)
        if os.path.isdir(subdir_path):
            convert_all_audio_in_dir(subdir_path)
