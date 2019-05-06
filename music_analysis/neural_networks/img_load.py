import os
import imageio
import numpy as np

import config

def get_img_filepaths(mel_spec_root_dir):
    '''
    Get the full filepaths to all images from a mel spec root dir
    '''
    img_paths = []

    for subdir in os.listdir(mel_spec_root_dir):
        subdir_path = os.path.join(mel_spec_root_dir, subdir)
        for img_file in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, img_file)
            img_paths.append(img_path)

    return img_paths

def track_id_to_str(track_id):
    t_str = str(track_id)
    zero_pad =  6 -len(t_str)
    return '0' * zero_pad + t_str

def track_id_to_filepath(track_id):
    track_id_str = track_id_to_str(track_id)
    subdir = os.path.join(config.MEL_SPEC_DIR, track_id_str[0:3])
    filepath = os.path.join(subdir, track_id_str + '.png')
    if not os.path.exists(filepath):
        # print('File: ', filepath, ' does not exist. :(')
        return None
    return filepath

def load_and_encode_img(track_id):
    filepath = track_id_to_filepath(track_id)
    if not filepath:
        return np.zeros(config.IMG_DIMS)

    image = imageio.imread(filepath)
    image = image.astype(np.float32)

    # Scale image channels to range [-1,1]
    image = (image - config.IMG_DEPTH // 2) / (config.IMG_DEPTH // 2)
    return image

# For testing (DEV)
if __name__ == '__main__':
    mel_spec_root = config.MEL_SPEC_DIR
    img_filepaths = get_img_filepaths(mel_spec_root)
    image = imageio.imread(img_filepaths[0])
    print(image.shape)

    image = load_and_encode_img(138060)
    print(image)