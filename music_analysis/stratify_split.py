import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold


X = pd.read_excel('essentia_features_WITH_LABEL.xlsx', sheet_name='Sheet1', usecols='B:PU', userows='2:1248')
y = pd.read_excel('essentia_features_WITH_LABEL.xlsx', sheet_name='Sheet1', usecols='PV', userows='2:1248')
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
sss.get_n_splits(X, y)

for train_index, test_index in sss.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train_init, X_test_final = X.iloc[train_index], X.iloc[test_index]
    y_train_init, y_test_final = y.iloc[train_index], y.iloc[test_index]

skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X_train_init, y_train_init)
for train_index, test_index in skf.split(X_train_init, y_train_init):
    # TODO: Model training here
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_train_init.iloc[train_index], X_train_init.iloc[test_index]
    y_train, y_test = y_train_init.iloc[train_index], y_train_init.iloc[test_index]





