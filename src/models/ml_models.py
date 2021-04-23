import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
import matplotlib.pyplot as plt

import os

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in


class ModelAttributes:
    LOGISTIC_REGRESSION = "logistic_regression_{}"
    NEURAL_NETWORK = "neural_network_{}"
    SVM = "svm_{}"
    DECISTION_TREE = "decision_tree_{}"
    RANDOM_FOREST = "random_forest_{}"
    MODEL_LOCATION = script_dir + "/updated_models/{}"
    MODEL_METRICS_LOCATION = script_dir + "/updated_metrics/{}"
    EMA_CLASSIFICATION = "ema_classification_{}"
    INCREASE_DECREASE_CLASSIFICATION = "increase_decrease_classification_{}"
    FEATURE_PLOT_DIRECTORY = script_dir + '/feature_plots/{}'
    TEST_SIZE = 0.33


def generate_models(data, x_col, y_col, model_type):
    logistic_regression_model(data, x_col, y_col, model_type)
    decision_tree_model(data, x_col, y_col, model_type)
    random_forest_model(data, x_col, y_col, model_type)
    return


# Logistic Regression
def logistic_regression_model(data, x_col, y_col, model_type):
    X_train, X_test, y_train, y_test = separate_data(data, x_col, y_col)

    lr = LogisticRegression()
    lr = lr.fit(X_train, y_train)
    model_name = ModelAttributes.LOGISTIC_REGRESSION.format(model_type)
    evaluate_model(lr, X_test, y_test, model_name)
    save_model(lr, model_name, ModelAttributes.MODEL_LOCATION)
    return


def decision_tree_model(data, x_col, y_col, model_type):
    X_train, X_test, y_train, y_test = separate_data(data, x_col, y_col)

    decision_tree = DecisionTreeClassifier()
    decision_tree = decision_tree.fit(X_train, y_train)

    model_name = ModelAttributes.DECISTION_TREE.format(model_type)
    evaluate_model(decision_tree, X_test, y_test, model_name)
    save_model(decision_tree, model_name, ModelAttributes.MODEL_LOCATION)
    return

def random_forest_model(data, x_col, y_col, model_type):
    X_train, X_test, y_train, y_test = separate_data(data, x_col, y_col)

    random_forest = RandomForestClassifier(max_depth=3, random_state=0, n_estimators=500)
    random_forest = random_forest.fit(X_train, y_train)
    random_forest_feature_importance(random_forest, x_col, model_type)
    model_name = ModelAttributes.RANDOM_FOREST.format(model_type)
    evaluate_model(random_forest, X_test, y_test, model_name)
    save_model(random_forest, model_name, ModelAttributes.MODEL_LOCATION)
    return


def svm_model(data, x_col, y_col, model_type):
    X_train, X_test, y_train, y_test = separate_data(data, x_col, y_col)

    svm = SVC(kernel='linear')  # kernel might need to be changed
    svm.fit(X_train, y_train)

    model_name = ModelAttributes.SVM.format(model_type)
    evaluate_model(svm, X_test, y_test, model_name)
    save_model(svm, model_name, ModelAttributes.MODEL_LOCATION)
    return


def neural_net_model(data, x_col, y_col, model_type):
    X_train, X_test, y_train, y_test = separate_data(data, x_col, y_col)

    # Create a model with keras
    # create and fit model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(12, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    # fit the keras model on the dataset
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=12)

    model_name = ModelAttributes.NEURAL_NETWORK.format(model_type)
    evaluate_model(model, X_test, y_test, model_name)
    # Saving the model requires to generate json first. keras does not work directly with pickle
    save_model(model.to_json(), model_name, ModelAttributes.MODEL_LOCATION)
    return


def separate_data(data, x_col, y_col):
    X = data[x_col]
    y = data[y_col]
    X = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ModelAttributes.TEST_SIZE)
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    model_metrics(actual=y_test, prediction=y_pred, model_name=model_name)


def model_metrics(actual, prediction, model_name):
    # generate precision, recall, accuracy and f1 score for the models. return object
    precision = precision_score(y_true=actual, y_pred=prediction)
    recall = recall_score(y_true=actual, y_pred=prediction)
    accuracy = accuracy_score(y_true=actual, y_pred=prediction)
    f1 = f1_score(y_true=actual, y_pred=prediction)

    metrics = [[precision, recall, accuracy, f1]]
    metrics_df = pd.DataFrame(metrics, columns=['precision', 'recall', 'accuracy', 'f1'])
    metrics_df.to_csv(ModelAttributes.MODEL_METRICS_LOCATION.format(model_name) + ".csv")
    return

def random_forest_feature_importance(model, x_col, model_type):
    # plotting feature importances
    features = x_col
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 15))
    plt.title('Feature Importance: {}'.format(model_type))
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig(ModelAttributes.FEATURE_PLOT_DIRECTORY.format(model_type + "_feature_plot"))
    plt.show()

def save_model(model, model_name, location):
    file = open(location.format(model_name), 'wb')
    pickle.dump(model, file)
    file.close()
    return
