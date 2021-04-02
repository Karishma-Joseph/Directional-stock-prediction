from unittest import TestCase
from models import *
from stock import Metrics
import pandas as pd
import os

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

test_data = pd.read_csv("../src/data/test.csv")
x_col = [Metrics.RSI, Metrics.MACD_DECISION, Metrics.OBV]
y_col = [Metrics.INCREASE_DECREASE.format(Metrics.CLOSE_PRICE)]

y_pred = [0, 1, 1, 0, 1, 0]
y_actual = [0, 0, 1, 0, 0, 0]

class Test(TestCase):
    def test_logistic_regression_model(self):
        logistic_regression_model(test_data, x_col=x_col, y_col=y_col, interval=Metrics.ONE_DAY)
        path_model = "../src/models/saved_models/{}".format(ModelAttributes.LOGISTIC_REGRESSION.format(Metrics.ONE_DAY))
        path_metrics = "../src/models/model_metrics/{}".format(ModelAttributes.LOGISTIC_REGRESSION.format(Metrics.ONE_DAY + ".csv"))
        self.assertTrue(os.path.isfile(path_model))
        self.assertTrue(os.path.isfile(path_metrics))
        os.remove(path_model)
        os.remove(path_metrics)
        return

    def test_decision_tree_model(self):
        decision_tree_model(test_data, x_col=x_col, y_col=y_col, interval=Metrics.ONE_DAY)
        path_model = "../src/models/saved_models/{}".format(ModelAttributes.DECISTION_TREE.format(Metrics.ONE_DAY))
        path_metrics = "../src/models/model_metrics/{}".format(
            ModelAttributes.DECISTION_TREE.format(Metrics.ONE_DAY + ".csv"))
        self.assertTrue(os.path.isfile(path_model))
        self.assertTrue(os.path.isfile(path_metrics))
        os.remove(path_model)
        os.remove(path_metrics)
        return

    def test_svm_model(self):
        svm_model(test_data, x_col=x_col, y_col=y_col, interval=Metrics.ONE_DAY)
        path_model = "../src/models/saved_models/{}".format(ModelAttributes.SVM.format(Metrics.ONE_DAY))
        path_metrics = "../src/models/model_metrics/{}".format(
            ModelAttributes.SVM.format(Metrics.ONE_DAY + ".csv"))
        self.assertTrue(os.path.isfile(path_model))
        self.assertTrue(os.path.isfile(path_metrics))
        os.remove(path_model)
        os.remove(path_metrics)
        return

    def test_neural_net_model(self):
        neural_net_model(test_data, x_col=x_col, y_col=y_col, interval=Metrics.ONE_DAY)
        path_model = "../src/models/saved_models/{}".format(ModelAttributes.NEURAL_NETWORK.format(Metrics.ONE_DAY))
        path_metrics = "../src/models/model_metrics/{}".format(
            ModelAttributes.NEURAL_NETWORK.format(Metrics.ONE_DAY + ".csv"))
        self.assertTrue(os.path.isfile(path_model))
        self.assertTrue(os.path.isfile(path_metrics))
        os.remove(path_model)
        os.remove(path_metrics)
        return

    def test_separate_data(self):
        separate_data(test_data, x_col=x_col, y_col=y_col)

    def test_evaluate_model(self):
        self.fail()

    def test_model_metrics(self):
        model_metrics(y_actual, y_pred, "test_model")

    def test_save_model(self):
        self.fail()

    def test_model_attributes(self):
        self.fail()
