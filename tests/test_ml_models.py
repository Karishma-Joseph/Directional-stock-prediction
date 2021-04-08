from unittest import TestCase
import pandas as pd
import os
from src.models import separate_data, logistic_regression_model, ModelAttributes, decision_tree_model, \
    random_forest_model, svm_model, neural_net_model, model_metrics
from src.stock import Metrics

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

# test_data = pd.read_csv("../src/data/test_data/aapl_1440_2017-01-01_2019-02-01.csv")
test_data = pd.read_csv("../src/data/training_data/aapl_5_2017-07-12_2019-02-01.csv")


col_ema_name = Metrics.EMA.format(Metrics.CLOSE_PRICE, Metrics.SPAN)
col_derivative = Metrics.DERIVATIVE.format(col_ema_name)
col_distance = Metrics.DISTANCE.format(col_ema_name, col_derivative)
crossover_count = Metrics.CROSSOVER_COUNT.format(Metrics.EMA.format(Metrics.CLOSE_PRICE, Metrics.SPAN))

x_col = x_col = [Metrics.RSI, Metrics.MACD_DECISION, Metrics.OBV, col_derivative, col_distance,
             crossover_count, Metrics.EPS, Metrics.REVENUE, Metrics.PE, Metrics.DEBT_EQUITY, Metrics.PRICE_TO_BOOK]
# y_col = [Metrics.INCREASE_DECREASE.format(Metrics.CLOSE_PRICE)]
y_col = ['Close_ema_50_classification']

y_pred = [0, 1, 1, 0, 1, 0]
y_actual = [0, 0, 1, 0, 0, 0]

class Test(TestCase):
    def test_logistic_regression_model(self):
        logistic_regression_model(test_data, x_col=x_col, y_col=y_col, interval=Metrics.ONE_DAY)
        path_model = "../src/models/saved_models/{}".format(ModelAttributes.LOGISTIC_REGRESSION.format(Metrics.ONE_DAY))
        path_metrics = "../src/models/model_metrics/{}".format(ModelAttributes.LOGISTIC_REGRESSION.format(str(Metrics.ONE_DAY) + ".csv"))
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

    def test_random_forest_model(self):
        model_type = ModelAttributes.EMA_CLASSIFICATION.format("apple" + "_" + str(Metrics.FIVE_MIN))

        random_forest_model(test_data, x_col=x_col, y_col=y_col, model_type=model_type)

        path_model = "../src/models/saved_models/{}".format(ModelAttributes.RANDOM_FOREST.format(Metrics.ONE_DAY))
        path_metrics = "../src/models/model_metrics/{}".format(
            ModelAttributes.RANDOM_FOREST.format(Metrics.ONE_DAY + ".csv"))
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
        path_metrics = "../src/models/model_metrics/{}".format(
            "test_model.csv")
        self.assertTrue(os.path.isfile(path_metrics))
        os.remove(path_metrics)


    def test_save_model(self):
        self.fail()

    def test_model_attributes(self):
        self.fail()
