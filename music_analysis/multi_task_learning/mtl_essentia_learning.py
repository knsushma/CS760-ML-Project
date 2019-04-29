from sklearn import preprocessing
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Dense
import keras
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold


def getFeatures(whole_df, part_df):
    df = whole_df.loc[whole_df['track_id'].isin(part_df['track_id'])]
    df = df.reset_index(drop=True)

    X = df.iloc[:,1:436]
    names = X.columns

    # standardizing X values
    scaler = preprocessing.StandardScaler()
    scaled_X = scaler.fit_transform(X)
    scaled_X = pd.DataFrame(scaled_X, columns=names)

    y_genre = df['Top Genre']

    filter_col = [col for col in df if col.startswith('Top Genre_')]
    one_hot_y_genre = df[filter_col].values

    filter_col = [col for col in df if col.startswith('Artist_')]
    one_hot_y_artist = df[filter_col].values

    #standardizing year
    scaled_y_year = scaler.fit_transform(df['Year'].values.reshape(-1,1))

    return df, scaled_X.values, y_genre, one_hot_y_genre, one_hot_y_artist, scaled_y_year






def transform(df):

    # filling na values for year with mean
    df['Year'] = df['Year'].fillna(df['Year'].mean())
    # dropping na values for artist
    df = df.dropna()
    df = df.reset_index(drop=True)


    # one-hot encoding genre
    one_hot_genre_df = pd.get_dummies(df['Top Genre'], prefix='Top Genre')
    one_hot_artist_df = pd.get_dummies(df['Artist'], prefix='Artist')

    df = pd.concat([df, one_hot_genre_df], axis=1)
    df = pd.concat([df, one_hot_artist_df], axis=1)

    return df


def run_model(X_train, X_test, y_genre_one_hot_train, y_artist_one_hot_train, y_year_train, y_genre_one_hot_test, y_artist_one_hot_test, y_year_test, epochs, batch_size):

    inputs = Input(shape=(X_train.shape[1],))

    # hidden layers(using only 1 currently)
    hid1 = Dense(1000, activation='relu')(inputs)
    # hid2 = Dense(1000, activation='relu')(hid1)
    # hid3 = Dense(1000, activation='relu')(hid2)
    # hid4 = Dense(1000, activation='relu')(hid3)

    genre_output = Dense(y_genre_one_hot_train.shape[1], activation='softmax', name='genre_output')(hid1)
    year_output = Dense(1, activation='linear', name='year_output')(hid1)
    artist_output = Dense(y_artist_one_hot_train.shape[1], activation='softmax', name='artist_output')(hid1)

    model = Model(inputs=inputs, outputs=[genre_output, artist_output, year_output])

    model.compile(optimizer=keras.optimizers.Adam(lr=0.01),
                  loss={'genre_output': 'categorical_crossentropy', 'artist_output': 'categorical_crossentropy',
                        'year_output': 'mean_squared_error'},
                  metrics={'genre_output': 'accuracy', 'artist_output': 'accuracy', 'year_output': 'mse'})

    model.fit(X_train, {'genre_output': y_genre_one_hot_train, 'artist_output': y_artist_one_hot_train,
                        'year_output': y_year_train}, epochs=epochs, verbose=1, batch_size=batch_size)

    loss_and_metrics = model.evaluate(X_test,
                                      {'genre_output': y_genre_one_hot_test, 'artist_output': y_artist_one_hot_test,
                                       'year_output': y_year_test}, verbose=1, batch_size=batch_size)

    '''
    for i, attr in enumerate(model.metrics_names):
        print(attr + ": ", loss_and_metrics[i])
    '''
    predictions = model.predict(X_test, batch_size=batch_size)
    return model, loss_and_metrics
    #print("done")


def run_model_caller(trainFile, testFile, epochs, batch_size, i=0):
    df_train = pd.read_csv(trainFile, header=0)
    df_test = pd.read_csv(testFile, header=0)
    df_train, X_train, y_genre_train, y_genre_one_hot_train, y_artist_one_hot_train, y_year_train = getFeatures(transformed_df, df_train)
    df_test, X_test, y_genre_test, y_genre_one_hot_test, y_artist_one_hot_test, y_year_test = getFeatures(transformed_df, df_test)

    model, loss_and_metrics = run_model(X_train=X_train, X_test=X_test, y_genre_one_hot_train=y_genre_one_hot_train,
                                        y_artist_one_hot_train=y_artist_one_hot_train, y_year_train=y_year_train,
                                        y_genre_one_hot_test=y_genre_one_hot_test,
                                        y_artist_one_hot_test=y_artist_one_hot_test, y_year_test=y_year_test,
                                        batch_size=batch_size, epochs=epochs)

    if i == 0:
        print("Model-evaluation:")
    else:
        print("Cross-validation attempt:" + str(i))

    for i, attr in enumerate(model.metrics_names):
        print(attr + ": ", loss_and_metrics[i])
    print("-------------------------------------")


if __name__ == "__main__":

    epochs = 30
    batch_size = 500
    whole_df = pd.read_csv('essentia_features_with_labels.csv', header=0)
    transformed_df = transform(whole_df)
    #part_df = pd.read_csv('essentia_test.csv', header=0)
    #y = whole_df.loc[whole_df['track_id'].isin(part_df['track_id'])]


    for i in range(1, 6):
        run_model_caller('essentia_trainfold_'+str(i)+'.csv', 'essentia_testfold_'+str(i)+'.csv', epochs, batch_size, i)

    run_model_caller('essentia_train.csv', 'essentia_test.csv', epochs, batch_size)







