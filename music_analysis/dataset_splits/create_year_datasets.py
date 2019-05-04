import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
import numpy as np


if __name__ == "__main__":

    filename = "essentia_features_with_labels.csv"

    df = pd.read_csv(filename, header=0)

    # First column just has numbers in ascending order... so removing them
    df = df.iloc[:, 1:]

    # Dropping na values for year
    df = df.dropna()
    df = df.reset_index(drop=True)

    # Stratification

    # TRYING OUT BINARY CLASSIFICATION FIRST
    # Below 2011 : 0, Equal or Above 2011: 1
    # df['Year classification label'] = np.where(df['Year'] >= 2011, 1, 0)

    # Multiple bins : 5 bins

    conditions = [
        (df['Year'] <= 2009),
        (df['Year'] > 2009) & (df['Year'] <= 2011),
        (df['Year'] > 2011) & (df['Year'] <= 2012),
        (df['Year'] > 2012) & (df['Year'] <= 2013),
        (df['Year'] > 2013)]

    choices = ['0', '1', '2', '3', '4']
    df['Year classification label'] = np.select(conditions, choices)

    # Stratifying is possible only if there are at least 2 samples for a given class
    g = df.groupby('Year classification label')
    df = g.filter(lambda x: len(x) > 1)

    X = df.iloc[:, :-5]
    y_genre_year = df['Year classification label']

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    sss.get_n_splits(X, y_genre_year)

    for train_indices, test_indices in sss.split(X, y_genre_year):
        df_train, df_test = df.iloc[train_indices], df.iloc[test_indices]
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

    # Write to csv
    df_train.to_csv('year_essentia_train.csv', index=True)
    df_test.to_csv('year_essentia_test.csv', index=True)
