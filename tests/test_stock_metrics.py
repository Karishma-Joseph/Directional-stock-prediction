from unittest import TestCase
import pandas as pd
import os

from src.stock import insert_data

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

test_data = pd.read_csv("../src/data/test_data/aapl_1440_2017-01-01_2019-02-01.csv", index_col=0)
new_data = pd.read_csv("../src/data/test_data/debt_equity.csv", index_col=0)


class Test(TestCase):
    def test_insert_data(self):
        insert_data(test_data, new_data, "temp")

