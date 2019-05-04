import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold


if __name__ == "__main__":

    filename = "essentia_features_with_labels.csv"

    df = pd.read_csv(filename, header=0)

    # First column just has numbers in ascending order... so removing them
    df= df.iloc[:, 1:]

    # Dropping na values for year
    df = df.dropna()
    df = df.reset_index(drop=True)

    # Stratification
    # TODO: Stratify based on sections of year

    # Stratifying is possible only if there are atleast 2 samples for a given class
    g = df.groupby('Year')
    df = g.filter(lambda x: len(x) > 1)

    X = df.iloc[:, :-5]
    y_genre_year = df['Year']
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    sss.get_n_splits(X, y_genre_year)

    for train_indices, test_indices in sss.split(X, y_genre_year):
        df_train, df_test = df.iloc[train_indices], df.iloc[test_indices]
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        # df_train, df_test = df.iloc[: -500, :], df.iloc[-500:, :]
        # df_train = df_train.reset_index(drop=True)
        # df_test = df_test.reset_index(drop=True)

    # Write to csv
    df_train.to_csv('year_essentia_train.csv', index=True)
    df_test.to_csv('year_essentia_test.csv', index=True)
