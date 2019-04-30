import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# Function to perform training with svm LinearSVC.
def train_using_svm_SVC(X_train, y_train):
    # Creating the classifier object
    svm_model = svm.SVC(kernel='rbf',gamma='auto')
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
          accuracy_score(y_test, y_pred))


# Function to make predictions
def prediction(X_test, clf_object):
    # Predicton
    y_pred = clf_object.predict(X_test)
    return y_pred


def standardize(X_train, X_test):
    # Standardize train and test data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def label_encoding(y_train, y_test):
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    return y_train, y_test


for i in range(1, 6):
    # Train data
    X_train = pd.read_csv('dataset_splits/essentia_trainfold_' + str(i) + '.csv')
    y_train = X_train.iloc[:, -2]
    X_train = X_train.iloc[:, 1:-6]
    # Test data
    X_test = pd.read_csv('dataset_splits/essentia_testfold_' + str(i) + '.csv')
    y_test = X_test.iloc[:, -2]
    X_test = X_test.iloc[:, 1:-6]

    # Standardize train and test data
    X_train, X_test = standardize(X_train, X_test)

    # Label Encoding labels
    y_train, y_test = label_encoding(y_train, y_test)

    # Model training
    svm_model = train_using_svm_SVC(X_train, y_train)
    clf_gini = train_using_gini(X_train, y_train)
    lr_model = train_using_logistic_regression(X_train, y_train)

    # Verify with test set

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

    print("********************************")

# Final test data

X_train = pd.read_csv('dataset_splits/essentia_train.csv')
y_train = X_train.iloc[:, -2]
X_train = X_train.iloc[:, 1:-6]

X_test = pd.read_csv('dataset_splits/essentia_test.csv')
y_test = X_test.iloc[:, -2]
X_test = X_test.iloc[:, 1:-6]

# Standardize train and test data
X_train, X_test = standardize(X_train, X_test)

# Label Encoding labels
y_train, y_test = label_encoding(y_train, y_test)

# Verify with test set

print("Final Results Using SVM:")
# Prediction Using SVM
y_pred_svm = prediction(X_test, svm_model)
cal_accuracy(y_test, y_pred_svm)

print("Final Results Using Logistic:")
# Prediction Using SVM
y_pred_lr = prediction(X_test, lr_model)
cal_accuracy(y_test, y_pred_lr)

print("Final Results Using GINI:")
# Prediction using gini
y_pred_gini = prediction(X_test, clf_gini)
cal_accuracy(y_test, y_pred_gini)
