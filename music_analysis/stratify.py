import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


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


# Function to perform training with randomForest.
def train_using_rf(X_train, y_train):
    # Creating the classifier object
    rf_model = RandomForestClassifier(n_estimators=30, max_depth=2,random_state=0)
    # Performing training
    rf_model.fit(X_train, y_train)
    return rf_model


# Function to perform training with logistic regression.
def train_using_logistic_regression(X_train, y_train):
    # Creating the classifier object
    lr_model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    # Performing training
    lr_model.fit(X_train, y_train)
    return lr_model


# Function to perform training with linear regression.
def train_using_linear_regression(X_train, y_train):
    # Creating the Linear Regression object
    linear_regression_model = LinearRegression()
    # Performing training
    linear_regression_model.fit(X_train, y_train)
    return linear_regression_model


# Function to calculate classification accuracy
def cal_classification_accuracy(y_test, y_pred):
    print("Accuracy : ",
          accuracy_score(y_test, y_pred))


# Function to calculate regression accuracy
def cal_regression_accuracy(y_test, y_pred):
    print("Accuracy using MSE: ",
          mean_squared_error(y_test, y_pred))
    print("Accuracy using R2 score: ",
          r2_score(y_test, y_pred))


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


def print_res_classification(X_test,svm_model,lr_model,rf_model,y_test):

    print("Results Using SVM:")
    # Prediction Using SVM
    y_pred_svm = prediction(X_test, svm_model)
    cal_classification_accuracy(y_test, y_pred_svm)

    print("Results Using Logistic:")
    # Prediction Using SVM
    y_pred_lr = prediction(X_test, lr_model)
    cal_classification_accuracy(y_test, y_pred_lr)

    print("Results Using Random Forest:")
    # Prediction using gini
    y_pred_gini = prediction(X_test, rf_model)
    cal_classification_accuracy(y_test, y_pred_gini)

    print("********************************")


def print_res_regression(X_test, linear_regression_model, y_test):
    print("Results Using Linear Regression:")
    # Prediction Using Linear Regression
    y_pred_linear_regression = prediction(X_test, linear_regression_model)
    # print("PREDICTED VALUES: \n")
    # print(y_pred_linear_regression)
    cal_regression_accuracy(y_test, y_pred_linear_regression)

    print("********************************")


if __name__ == "__main__":

    # Cross-validation TODO: Check if required. Modify accordingly.
    # 5 fold cross validation
    # no_folds = 5

    # for i in range(1, no_folds):
    #     # Train data for genre
    #     X_train = pd.read_csv('dataset_splits/essentia_trainfold_' + str(i) + '.csv')
    #     y_train_genre = X_train.iloc[:, -2]
    #     y_train_artist = X_train.iloc[:, -4]
    #     y_train_year = X_train.iloc[:, -3]
    #     y_train_year = y_train_year.round(0).astype(int)
    #     X_train = X_train.iloc[:, 1:-6]
    #     # Test data for genre
    #     X_test = pd.read_csv('dataset_splits/essentia_testfold_' + str(i) + '.csv')
    #     y_test_genre = X_test.iloc[:, -2]
    #     y_test_artist = X_test.iloc[:, -4]
    #     X_test = X_test.iloc[:, 1:-6]
    #
    #     # Standardize train and test data
    #     X_train, X_test = standardize(X_train, X_test)
    #
    #     # Label Encoding labels genre
    #     y_train_genre, y_test_genre = label_encoding(y_train_genre, y_test_genre)
    #
    #     # Model training for genre prediction
    #     svm_model_genre = train_using_svm_SVC(X_train, y_train_genre)
    #     lr_model_genre = train_using_logistic_regression(X_train, y_train_genre)
    #     rf_model_genre = train_using_rf(X_train, y_train_genre)
    #
    #     # Print test results for genre
    #     print("GENRE:")
    #     print_res_classification(X_test,svm_model_genre,lr_model_genre,rf_model_genre,y_test_genre)
    #
    #     # Label Encoding labels artist
    #     y_train_artist, y_test_artist = label_encoding(y_train_artist, y_test_artist)
    #
    #     print("ARTIST:")
    #     # Model training for artist prediction
    #     svm_model_artist = train_using_svm_SVC(X_train, y_train_artist)
    #     lr_model_artist = train_using_logistic_regression(X_train, y_train_artist)
    #     rf_model_artist = train_using_rf(X_train, y_train_artist)
    #     # Print test results for artist
    #     print_res_classification(X_test,svm_model_artist,lr_model_artist,rf_model_artist,y_test_artist)

    # Final test results

    # GENRE AND ARTIST

    X_train = pd.read_csv('dataset_splits/essentia_train.csv')
    y_train_genre = X_train.iloc[:, -2]
    y_train_artist = X_train.iloc[:, -4]
    X_train = X_train.iloc[:, 1:-6]

    X_test = pd.read_csv('dataset_splits/essentia_test.csv')
    y_test_genre = X_test.iloc[:, -2]
    y_test_artist = X_test.iloc[:, -4]
    X_test = X_test.iloc[:, 1:-6]

    # Standardize train and test data
    X_train, X_test = standardize(X_train, X_test)

    # Label Encoding labels
    y_train_genre, y_test_genre = label_encoding(y_train_genre, y_test_genre)

    # Model training for genre prediction
    svm_model_genre = train_using_svm_SVC(X_train, y_train_genre)
    lr_model_genre = train_using_logistic_regression(X_train, y_train_genre)
    rf_model_genre = train_using_rf(X_train, y_train_genre)

    # Print test results for genre
    print("GENRE:")
    print_res_classification(X_test, svm_model_genre,lr_model_genre, rf_model_genre, y_test_genre)

    # Label Encoding labels artist
    y_train_artist, y_test_artist = label_encoding(y_train_artist, y_test_artist)

    # Model training for artist prediction
    svm_model_artist = train_using_svm_SVC(X_train, y_train_artist)
    lr_model_artist = train_using_logistic_regression(X_train, y_train_artist)
    rf_model_artist = train_using_rf(X_train, y_train_artist)

    # Print test results for artist
    print("ARTIST:")
    print_res_classification(X_test, svm_model_artist,lr_model_artist,rf_model_artist, y_test_artist)

    # YEAR

    X_train = pd.read_csv('dataset_splits/year_essentia_train.csv')
    y_train_year = X_train.iloc[:, -3]
    y_train_year_classification_label = X_train.iloc[:, -1]
    X_train = X_train.iloc[:, 1:-6]

    X_test = pd.read_csv('dataset_splits/year_essentia_test.csv')
    y_test_year = X_test.iloc[:, -3]
    y_test_year_classification_label = X_test.iloc[:, -1]
    X_test = X_test.iloc[:, 1:-6]

    # Standardize train and test data
    X_train, X_test = standardize(X_train, X_test)

    # A. Regression

    # Model training for year regression
    linear_regression_model_year = train_using_linear_regression(X_train, y_train_year)
    # Print test results for Year regression
    print("UNSTRATIFIED YEAR (REGRESSION) : ")
    print_res_regression(X_test, linear_regression_model_year, y_test_year)

    # B. Classification

    # Model training for year classification
    svm_model_year = train_using_svm_SVC(X_train, y_train_year_classification_label)
    lr_model_year = train_using_logistic_regression(X_train, y_train_year_classification_label)
    rf_model_year = train_using_rf(X_train, y_train_year_classification_label)

    # Print test results for year classification
    print("STRATIFIED YEAR (CLASSIFICATION):")
    print_res_classification(X_test, svm_model_year,
                             lr_model_year, rf_model_year, y_test_year_classification_label)
