import labels_load
import img_load
import config

import numpy as np

class Loader:
    def __init__(self, label_csv_filepaths, batch_size, stochastic=True):
        self._labels = labels_load.load_labels(label_csv_filepaths[0])
        for filepath in label_csv_filepaths[1:]:
            new_labels = labels_load.load_labels(filepath)
            self._labels = self._labels.append(new_labels, ignore_index=True)

        self._batch_size = batch_size
        self._stochastic = stochastic

        self._curr_idx = 0
        self._max_idx = self._labels.shape[0]

        if self._stochastic:
            self._shuffle_rows()

    def next_batch(self):
        # Check for end of Epoch; if so, reset & return NULL
        if self._curr_idx >= self._max_idx:
            print('End of Epoch')
            self._curr_idx = 0
            if self._stochastic:
                self._shuffle_rows

            return None

        batch = self._labels.loc[self._curr_idx:self._curr_idx+self._batch_size-1, :]
        encoded_batch = self._encode_batch(batch)

        self._curr_idx += self._batch_size
        return encoded_batch

    def _shuffle_rows(self):
        self._labels = self._labels.sample(frac=1).reset_index(drop=True)

    def _encode_batch(self, batch):
        batch_size = batch.shape[0]
        
        # TODO: Implement encoding on artists, genres
        images = np.zeros([batch_size] + config.IMG_DIMS)
        years = np.zeros([batch_size] + [1])
        genres = np.zeros([batch_size] + [8])
        artists = np.zeros([batch_size] + [config.NUM_ARTISTS])

        for j, i in enumerate(range(self._curr_idx, self._curr_idx + batch_size)):
            images[j,:,:,:] = img_load.load_and_encode_img(batch.loc[i, 'track_id'])
            years[j,0] = batch.loc[i, 'Year']
            years[j,0] = (years[j,0] - config.MIN_YEAR) / (config.MAX_YEAR - config.MIN_YEAR)
            # if np.isnan(years[j,0]):
            #     years[j,0] = 2000
            genres[j, config.GENRES.index(batch.loc[i, 'Top Genre'])] = 1
            artists[j, config.ARTISTS.index(batch.loc[i, 'Artist'])] = 1

        return images, years, genres, artists

if __name__ == '__main__':
    test_loader = Loader([config.TRAIN_FOLDS_LABELS[0]], 8, stochastic=False)

    batch = test_loader.next_batch()
    while batch:
        batch = test_loader.next_batch()
        print(batch[3])