import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# Function to perform training with svm LinearSVC.
def train_using_svm_SVC(X_train, y_train):
    # Creating the classifier object
    svm_model = svm.SVC(kernel='rbf')
    # Performing training
    svm_model.fit(X_train, y_train)
    return svm_model

# Function to perform training with giniIndex.
def train_using_gini(X_train, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=100, max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

# Function to perform training with logistic regression.
def train_using_logistic_regression(X_train, y_train):
    # Creating the classifier object
    lr_model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    # Performing training
    lr_model.fit(X_train, y_train)
    return lr_model


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):

    print("Accuracy : ",
          accuracy_score(y_test,y_pred))


# Function to make predictions
def prediction(X_test, clf_object):
    # Predicton
    y_pred = clf_object.predict(X_test)
    return y_pred

X = pd.read_excel('essentia_features_WITH_LABEL.xlsx', sheet_name='Sheet1', usecols='B:PV', userows='2:7219')
X = X.dropna()
y = X.iloc[:,-1]
X = X.iloc[:,:-1]
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
sss.get_n_splits(X, y)

for train_index, test_index in sss.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train_init, X_test_final = X[train_index], X[test_index]
    y_train_init, y_test_final = y[train_index], y[test_index]

skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X_train_init, y_train_init)
for train_index, test_index in skf.split(X_train_init, y_train_init):

    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_train_init[train_index], X_train_init[test_index]
    y_train, y_test = y_train_init[train_index], y_train_init[test_index]

    #Model training

    svm_model = train_using_svm_SVC(X_train, y_train)
    clf_gini = train_using_gini(X_train, y_train)
    lr_model = train_using_logistic_regression(X_train, y_train)

    #Verify with test set

    print("Results Using SVM:")
    # Prediction Using SVM
    y_pred_svm = prediction(X_test, svm_model)
    cal_accuracy(y_test, y_pred_svm)

    print("Results Using Logistic:")
    # Prediction Using SVM
    y_pred_lr = prediction(X_test, lr_model)
    cal_accuracy(y_test, y_pred_lr)

    print("Results Using GINI:")
    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)




    