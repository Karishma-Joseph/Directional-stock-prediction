from unittest import TestCase
from models import *
from stock import Metrics
import pandas as pd

test_data = pd.read_csv("../src/data/test.csv")
x_col = [Metrics.RSI, Metrics.MACD_DECISION, Metrics.OBV]
y_col = [Metrics.INCREASE_DECREASE.format(Metrics.CLOSE_PRICE)]

y_pred = [0, 1, 1, 0, 1, 0]
y_actual = [0, 0, 1, 0, 0, 0]

class Test(TestCase):
    def test_logistic_regression_model(self):
        logistic_regression_model(test_data, x_col=x_col, y_col=y_col, interval=Metrics.ONE_DAY)

    def test_decision_tree_model(self):
        self.fail()

    def test_svm_model(self):
        self.fail()

    def test_neural_net_model(self):
        self.fail()

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
