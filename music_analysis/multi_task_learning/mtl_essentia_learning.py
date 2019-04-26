import keras.backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Dense
import keras

def load_csv(filename):

    df = pd.read_csv(filename, header=0)

    #shuffle values
    df = df.sample(frac=1, replace=False)


    #TODO: Deal with missing values
    df['Year'] = df['Year'].fillna(2000)
    df['Top Genre'] = df['Top Genre'].fillna('Experimental')
    #df['Artist'] = df['Artist'].fillna('AWOL')

    #dropping na values for artist
    df = df.dropna()

    df = df.reset_index(drop=True)


    X = df.iloc[:,1:-5]
    names = X.columns

    #standardizing X values
    scaler = preprocessing.StandardScaler()
    scaled_X = scaler.fit_transform(X)
    scaled_X = pd.DataFrame(scaled_X, columns=names)


    #one-hot encoding genre
    y_genre= df['Top Genre']
    le_genre = preprocessing.LabelEncoder()
    le_genre.fit(y_genre.values)
    one_hot_y_genre = keras.utils.to_categorical(le_genre.transform(y_genre.values))


    #one-hot encoding artist
    y_artist = df['Artist']
    le_artist = preprocessing.LabelEncoder()
    le_artist.fit(y_artist.values)
    one_hot_y_artist = keras.utils.to_categorical(le_artist.transform(y_artist.values))

    #standardizing year
    scaled_y_year = scaler.fit_transform(df['Year'].values.reshape(-1,1))


    return scaled_X.values,one_hot_y_genre, one_hot_y_artist, scaled_y_year






if __name__ == "__main__":
    X, y_genre, y_artist, y_year = load_csv(filename='essentia_features_with_labels.csv')

    epochs = 30
    batch_size = 500

    inputs = Input(shape=(X.shape[1],))

    #hidden layers(using only 1 currently)
    hid1 = Dense(1000, activation='relu')(inputs)
    #hid2 = Dense(1000, activation='relu')(hid1)
    #hid3 = Dense(1000, activation='relu')(hid2)
    #hid4 = Dense(1000, activation='relu')(hid3)


    genre_output = Dense(y_genre.shape[1], activation='softmax', name='genre_output')(hid1)
    year_output = Dense(1, activation='linear', name='year_output')(hid1)
    artist_output = Dense(y_artist.shape[1], activation='softmax', name='artist_output')(hid1)

    model = Model(inputs=inputs, outputs=[genre_output, artist_output, year_output])

    model.compile(optimizer=keras.optimizers.Adam(lr=0.01),
                  loss={'genre_output': 'categorical_crossentropy', 'artist_output': 'categorical_crossentropy', 'year_output': 'mean_squared_error'},
                  metrics={'genre_output':'accuracy', 'artist_output':'accuracy', 'year_output':'mse'})

    model.fit(X, {'genre_output' : y_genre, 'artist_output' : y_artist, 'year_output' : y_year}, epochs= epochs, verbose=1, batch_size=batch_size)






