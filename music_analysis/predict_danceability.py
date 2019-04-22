import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from numpy import genfromtxt


def predict_accuracy_on_diff_classifiers(data_set):
    print("Accuracy to be found using diff classifiers")
    cross_validation_fold = 10
    feature_set = data_set[:, 1:-1].astype(float)
    label_set = data_set[:, -1].astype(float)

    # # 1. Decision Tree
    # decision_tree_model = DecisionTreeClassifier()
    # cross_validation_score = cross_val_score(decision_tree_model,
    #                                             feature_set, label_set, cv=cross_validation_fold, scoring='roc_auc')
    # average_cross_validation_accuracy = np.mean(cross_validation_score)
    # print("1. Decision Tree accuracy: ", average_cross_validation_accuracy)
    #
    #
    # # 2. Random Forest
    # random_forest_model = RandomForestClassifier(n_estimators=100)
    # cross_validation_score = cross_val_score(
    #     random_forest_model, feature_set, label_set, cv=cross_validation_fold, scoring='roc_auc')
    # average_cross_validation_accuracy = np.mean(cross_validation_score)
    # print("2. Random Forest accuracy: ", average_cross_validation_accuracy)
    #
    # # 3. Support Vector Machine
    # svm_model = svm.SVC(kernel='linear', C=1)
    # cross_validation_score = cross_val_score(
    #     svm_model, feature_set, label_set, cv=cross_validation_fold, scoring='roc_auc')
    # average_cross_validation_accuracy = np.mean(cross_validation_score)
    # print("3. Support Vector Machine accuracy: ", average_cross_validation_accuracy)

    # 4. Linear Regression
    linear_regression_model = LinearRegression()
    cross_validation_accuracy = cross_val_score(
        linear_regression_model, feature_set, label_set, cv=cross_validation_fold, scoring='neg_mean_squared_error')
    average_cross_validation_accuracy = abs(np.mean(cross_validation_accuracy))
    print("4. Linear Regression accuracy: ", average_cross_validation_accuracy)

    # # 5. Logistic Regression
    # logistic_regression_model = LogisticRegression(solver='lbfgs')
    # cross_validation_accuracy = cross_val_score(logistic_regression_model, feature_set,
    #                                             label_set, cv=cross_validation_fold, scoring='roc_auc')
    # average_cross_validation_accuracy = np.mean(cross_validation_accuracy)
    # print("5. Logistic Regression accuracy: ", average_cross_validation_accuracy)


if __name__ == '__main__':
    csv_data = genfromtxt('train_set_music.csv', delimiter=',')
    predict_accuracy_on_diff_classifiers(csv_data)
