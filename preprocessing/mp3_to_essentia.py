import os
import config

def convert_all_audio_in_dir(subdir_path):
    dir_name = os.path.basename(os.path.normpath(subdir_path))
    for audio_file in os.listdir(subdir_path):
        audio_file_path = os.path.join(subdir_path, audio_file)

        essentia_dir_path = os.path.join(config.ESS_DIR, dir_name)
        if not os.path.exists(essentia_dir_path):
            os.makedirs(essentia_dir_path)
        essentia_file_path = os.path.join(essentia_dir_path, audio_file[:-3]+"json")
        
        #call essentia api for getting features
        command = "essentia_streaming_extractor_music "+audio_file_path+" "+essentia_file_path
        os.system(command)

if __name__ == "__main__":

    for subdir in os.listdir(config.AUDIO_DIR):
        subdir_path = os.path.join(config.AUDIO_DIR, subdir)
        if os.path.isdir(subdir_path):
            convert_all_audio_in_dir(subdir_path)
