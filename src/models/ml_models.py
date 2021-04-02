import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam


class model_attributes:
    LOGISTIC_REGRESSION = "logistic_regression_{}"
    NEURAL_NETWORK = "neural_network_{}"
    SVM = "svm_{}"
    DECISTION_TREE = "decision_tree_{}"
    MODEL_LOCATION = "/saved_models/{}"


# Logistic Regression
def logistic_regression_model(data, x_col, y_col, interval):
    # Required Column Names
    X = data[[x_col]]
    y = data[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    lr = LogisticRegression().fit(X_train, y_train)
    r_sq = lr.score(X_train, y_train)

    predictions = lr.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    model_name = model_attributes.LOGISTIC_REGRESSION.format(interval)
    save_model(lr, model_name, model_attributes.MODEL_LOCATION)
    return


def decision_tree_model(data, x_col, y_col, interval):
    X = data[[x_col]]
    y = data[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Create Decision Tree classifer object
    decision_tree = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    decision_tree = decision_tree.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = decision_tree.predict(X_test)
    model_name = model_attributes.DECISTION_TREE.format(interval)
    save_model(decision_tree, model_name, model_attributes.MODEL_LOCATION)
    return


def svm_model(data, x_col, y_col, interval):
    X = data[[x_col]]
    y = data[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Create a svm Classifier
    svm = SVC(kernel='linear')  # Linear Kernel

    # Train the model using the training sets
    svm.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = svm.predict(X_test)
    model_name = model_attributes.SVM.format(interval)
    save_model(svm, model_name, model_attributes.MODEL_LOCATION)
    return


def neural_net_model(data, x_col, y_col, interval):
    X = data[[x_col]]
    y = data[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Create a model with keras
    # create and fit the LSTM network
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(12, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1))

    model.summary()
    opt = Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    # fit the keras model on the dataset
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=12)
    model_name = model_attributes.NEURAL_NETWORK.format(interval)
    save_model(model, model_name, model_attributes.MODEL_LOCATION)
    return


def save_model(model, model_name, location):
    pickle.dump(model, open(location.format(model_name), 'wb'))
    return
