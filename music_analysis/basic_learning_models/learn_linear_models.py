import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
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


# Function to perform training with ridge regression.
def train_using_ridge_regression(X_train, y_train):
    # Creating the Ridge Regression object
    linear_ridge_model_year = Ridge()
    # Performing training
    linear_ridge_model_year.fit(X_train, y_train)
    return linear_ridge_model_year


# Function to perform training with lasso regression.
def train_using_lasso_regression(X_train, y_train):
    # Creating the Lasso Regression object
    lasso_regression_model = Lasso()
    # Performing training
    lasso_regression_model.fit(X_train, y_train)
    return lasso_regression_model


# Function to calculate classification accuracy
def cal_classification_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy : ", accuracy)
    return accuracy


# Function to calculate regression accuracy
def cal_regression_accuracy(y_test, y_pred):
    regression_mean_absolute_error = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error: ", regression_mean_absolute_error)
    return regression_mean_absolute_error


# Function to make predictions
def prediction(X_test, clf_object):
    # Prediction
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
    svm_accuracy_score = cal_classification_accuracy(y_test, y_pred_svm)

    print("Results Using Logistic:")
    # Prediction Using SVM
    y_pred_lr = prediction(X_test, lr_model)
    lr_accuracy_score = cal_classification_accuracy(y_test, y_pred_lr)

    print("Results Using Random Forest:")
    # Prediction using gini
    y_pred_gini = prediction(X_test, rf_model)
    rf_accuracy_score = cal_classification_accuracy(y_test, y_pred_gini)

    print("********************************")
    return svm_accuracy_score, lr_accuracy_score, rf_accuracy_score


def print_res_regression(X_test, linear_regression_model,
                         ridge_regression_model, lasso_regression_model, y_test):
    # Prediction
    print("Results Using Linear Regression:")
    y_pred_linear_regression = prediction(X_test, linear_regression_model)
    linear_regression_accuracy_score = cal_regression_accuracy(y_test, y_pred_linear_regression)

    print("Results Using Ridge Regression:")
    y_pred_ridge_regression = prediction(X_test, ridge_regression_model)
    ridge_regression_accuracy_score = cal_regression_accuracy(y_test, y_pred_ridge_regression)

    print("Results Using Lasso Regression:")
    y_pred_lasso_regression = prediction(X_test, lasso_regression_model)
    lasso_regression_accuracy_score = cal_regression_accuracy(y_test, y_pred_lasso_regression)

    print("********************************")
    return linear_regression_accuracy_score, ridge_regression_accuracy_score, lasso_regression_accuracy_score

def train_for_genre(X_train, X_test, y_train_genre, y_test_genre):
    # Standardize train and test data
    X_train, X_test = standardize(X_train, X_test)

    # Label Encoding labels genre
    y_train_genre, y_test_genre = label_encoding(y_train_genre, y_test_genre)

    # Model training for genre prediction
    svm_model_genre = train_using_svm_SVC(X_train, y_train_genre)
    lr_model_genre = train_using_logistic_regression(X_train, y_train_genre)
    rf_model_genre = train_using_rf(X_train, y_train_genre)

    # Print test results for genre
    print("GENRE:")
    return print_res_classification(X_test, svm_model_genre, lr_model_genre, rf_model_genre, y_test_genre)

def train_for_artist(X_train, X_test, y_train_artist, y_test_artist):
    # Label Encoding labels artist
    y_train_artist, y_test_artist = label_encoding(y_train_artist, y_test_artist)

    # Model training for artist prediction
    svm_model_artist = train_using_svm_SVC(X_train, y_train_artist)
    lr_model_artist = train_using_logistic_regression(X_train, y_train_artist)
    rf_model_artist = train_using_rf(X_train, y_train_artist)

    # Print test results for artist
    print("ARTIST:")
    return print_res_classification(X_test, svm_model_artist, lr_model_artist, rf_model_artist, y_test_artist)


def train_for_year(X_train, X_test, y_train_year, y_test_year):
    # standarddize labels for year
    y_train_year, y_test_year = standardize(y_train_year.values.reshape(-1, 1), y_test_year.values.reshape(-1, 1))

    # Model training for year regression
    linear_regression_model_year = train_using_linear_regression(X_train, y_train_year)
    # Print test results for Year regression
    print("UNSTRATIFIED YEAR (REGRESSION) : ")
    return print_res_regression(X_test, linear_regression_model_year, y_test_year)


def get_train_dataset(file_name):
    X_train = pd.read_csv(file_name)
    y_train_genre = X_train.iloc[:, -2]
    y_train_artist = X_train.iloc[:, -4]
    y_train_year = X_train.iloc[:, -3]
    # y_train_year = y_train_year.round(0).astype(int)
    X_train = X_train.iloc[:, 1:-6]
    return X_train, y_train_genre, y_train_artist, y_train_year

def get_test_dataset(file_name):
    X_test = pd.read_csv(file_name)
    y_test_genre = X_test.iloc[:, -2]
    y_test_artist = X_test.iloc[:, -4]
    y_test_year = X_test.iloc[:, -3]
    X_test = X_test.iloc[:, 1:-6]
    return X_test, y_test_genre, y_test_artist, y_test_year

if __name__ == "__main__":

    # GENRE AND ARTIST

    X_train = pd.read_csv('../dataset_splits/essentia_train.csv')
    y_train_genre = X_train.iloc[:, -2]
    y_train_year = X_train.iloc[:, -3]
    y_train_artist = X_train.iloc[:, -4]
    X_train = X_train.iloc[:, 1:-6]

    X_test = pd.read_csv('../dataset_splits/essentia_test.csv')
    y_test_genre = X_test.iloc[:, -2]
    y_test_year = X_test.iloc[:, -3]
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

    # A. Regression
    # Standardize year values
    y_train_year, y_test_year = standardize(y_train_year.values.reshape(-1, 1), y_test_year.values.reshape(-1, 1))

    # Model training for year regression
    linear_regression_model_year = train_using_linear_regression(X_train, y_train_year)
    linear_ridge_model_year = train_using_ridge_regression(X_train, y_train_year)
    linear_lasso_model_year = train_using_lasso_regression(X_train, y_train_year)
    # Print test results for Year regression
    print("UNSTRATIFIED YEAR (REGRESSION) : ")
    print_res_regression(X_test, linear_regression_model_year,
                         linear_ridge_model_year, linear_lasso_model_year, y_test_year)

    # B. Classification
    # Changing to the dataset with no missing values for year
    X_train = pd.read_csv('../dataset_splits/year_essentia_train.csv')
    y_train_year_classification_label = X_train.iloc[:, -1]
    X_train = X_train.iloc[:, 1:-6]

    X_test = pd.read_csv('../dataset_splits/year_essentia_test.csv')
    y_test_year_classification_label = X_test.iloc[:, -1]
    X_test = X_test.iloc[:, 1:-6]

    # Standardize train and test data
    X_train, X_test = standardize(X_train, X_test)

    # Model training for year classification
    svm_model_year = train_using_svm_SVC(X_train, y_train_year_classification_label)
    lr_model_year = train_using_logistic_regression(X_train, y_train_year_classification_label)
    rf_model_year = train_using_rf(X_train, y_train_year_classification_label)

    # Print test results for year classification
    print("STRATIFIED YEAR (CLASSIFICATION):")
    print_res_classification(X_test, svm_model_year,
                             lr_model_year, rf_model_year, y_test_year_classification_label)

    # # CROSS-VALIDATION: 5 fold cross validation
    # no_folds = 5
    #
    # genre_svm_accuracy_score = 0.0
    # genre_lr_accuracy_score = 0.0
    # genre_rf_accuracy_score = 0.0
    # artist_svm_accuracy_score = 0.0
    # artist_lr_accuracy_score = 0.0
    # artist_rf_accuracy_score = 0.0
    # year_lr_accuracy_score = 0.0
    #
    # for i in range(1, no_folds+1):
    #     # Train dataset
    #     X_train, y_train_genre, y_train_artist, y_train_year = get_train_dataset('../dataset_splits/essentia_trainfold_' + str(i) + '.csv')
    #
    #     # Test dataset
    #     X_test, y_test_genre, y_test_artist, y_test_year = get_test_dataset('../dataset_splits/essentia_testfold_' + str(i) + '.csv')
    #
    #     # Train for Genre
    #     svm_accuracy_score_genre, lr_accuracy_score_genre, rf_accuracy_score_genre = train_for_genre(X_train, X_test, y_train_genre, y_test_genre)
    #     genre_svm_accuracy_score += svm_accuracy_score_genre
    #     genre_lr_accuracy_score += lr_accuracy_score_genre
    #     genre_rf_accuracy_score += rf_accuracy_score_genre
    #
    #     # Train for Artist
    #     svm_accuracy_score_artist, lr_accuracy_score_artist, rf_accuracy_score_artist = train_for_artist(X_train, X_test, y_train_artist, y_test_artist)
    #     artist_svm_accuracy_score += svm_accuracy_score_artist
    #     artist_lr_accuracy_score += lr_accuracy_score_artist
    #     artist_rf_accuracy_score += rf_accuracy_score_artist
    #
    #     # Train for Year
    #     lr_accuracy_score_year = train_for_year(X_train, X_test, y_train_year, y_test_year)
    #     year_lr_accuracy_score += lr_accuracy_score_year
    #
    # # Accurate/Best Model Analysis
    # genre_svm_accuracy_score /= no_folds
    # genre_lr_accuracy_score /= no_folds
    # genre_rf_accuracy_score /= no_folds
    # artist_svm_accuracy_score /= no_folds
    # artist_lr_accuracy_score /= no_folds
    # artist_rf_accuracy_score /= no_folds
    # year_lr_accuracy_score /= no_folds
    # print("Genre Accuracy: SVM = {0} | LR = {1} | RF =
    # {2}".format(genre_svm_accuracy_score, genre_lr_accuracy_score, genre_rf_accuracy_score))
    # print("Artist Accuracy: SVM = {0} | LR = {1} | RF =
    # {2}".format(artist_svm_accuracy_score, artist_lr_accuracy_score, artist_rf_accuracy_score))
    # print("Genre Accuracy: LR = {0}".format(year_lr_accuracy_score))
