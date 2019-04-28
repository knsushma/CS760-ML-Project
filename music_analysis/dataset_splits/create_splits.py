import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold


if __name__ == "__main__":

    filename= "essentia_features_with_labels.csv"

    df = pd.read_csv(filename, header=0)

    #first column just has numbers in ascending order... so removing them
    df= df.iloc[:, 1:]

    #filling na values for year with mean
    df['Year'] = df['Year'].fillna(df['Year'].mean())


    # dropping na values for artist
    df = df.dropna()
    df = df.reset_index(drop=True)


    #concatenating top_genre, artist as a clever means for stratifying based on 2 columns
    df['Genre and Artist'] = df['Top Genre'].astype(str) + df['Artist'].astype(str)

    #stratifying is possible only if there are atleast 2 samples for a given class
    g = df.groupby('Genre and Artist')
    df = g.filter(lambda x: len(x) > 1)

    X = df.iloc[:, :-5]
    y_genre_artist = df['Genre and Artist']
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    sss.get_n_splits(X, y_genre_artist)

    for train_indices, test_indices in sss.split(X, y_genre_artist):
        df_train, df_test = df.iloc[train_indices], df.iloc[test_indices]
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        df_train.to_csv('essentia_train.csv', index=True)
        df_test.to_csv('essentia_test.csv', index=True)

    skf = StratifiedKFold(n_splits=5)

    X_train = df_train.iloc[:, :-5]
    y_genre_artist_train = df_train['Genre and Artist']

    skf.get_n_splits(X_train, y_genre_artist_train)

    for i, (train_indices, test_indices) in enumerate(skf.split(X_train, y_genre_artist_train)):
        df_train_fold, df_test_fold = df_train.iloc[train_indices], df_train.iloc[test_indices]
        df_train_fold = df_train_fold.reset_index(drop=True)
        df_test_fold = df_test_fold.reset_index(drop=True)

        df_train_fold.to_csv('essentia_trainfold_'+str(i+1)+'.csv')
        df_test_fold.to_csv('essentia_testfold_'+str(i+1)+'.csv')
