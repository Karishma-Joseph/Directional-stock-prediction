from stock import Stock, Metrics
from models import *
import os
import re

intervals = [Metrics.FIVE_MIN, Metrics.FIFTEEN_MIN, Metrics.THIRTY_MIN, Metrics.ONE_HOUR, Metrics.FOUR_HOUR,
             Metrics.ONE_DAY]


def generate_all_models():
    companies = ['aapl', 'amzn']

    col_ema_name = Metrics.EMA.format(Metrics.CLOSE_PRICE, Metrics.SPAN)
    col_derivative = Metrics.DERIVATIVE.format(col_ema_name)
    col_distance = Metrics.DISTANCE.format(col_ema_name, col_derivative)
    crossover_count = Metrics.CROSSOVER_COUNT.format(Metrics.EMA.format(Metrics.CLOSE_PRICE, Metrics.SPAN))

    x_col = [Metrics.RSI, Metrics.MACD_DECISION, Metrics.OBV, col_derivative, col_distance,
             crossover_count, Metrics.EPS, Metrics.REVENUE, Metrics.PE, Metrics.DEBT_EQUITY, Metrics.PRICE_TO_BOOK]
    y_cols = [Metrics.INCREASE_DECREASE.format(Metrics.CLOSE_PRICE), 'Close_ema_50_classification']
    for y_col in y_cols:
        for company in companies:
            for file in os.listdir("data/training_data/"):
                if company in file:
                    data = pd.read_csv("data/training_data/{}".format(file))
                    interval = re.search('(\d+)', file).group(0)
                    if "classification" in y_col:
                        model_type = ModelAttributes.EMA_CLASSIFICATION.format(company + "_" + str(interval))
                    else:
                        model_type = ModelAttributes.INCREASE_DECREASE_CLASSIFICATION.format(company + "_" + str(interval))
                    generate_models(data=data, x_col=x_col, y_col=y_col, model_type=model_type)


def generate_all_data():
    start_dates_amazon = ['2017-11-09', '2017-09-25', '2017-05-12', '2017-03-13', '2015-11-27', '2014-09-29']
    start_dates_apple = ['2017-07-12', '2016-12-22', '2016-08-23', '2016-03-16', '2014-06-30', '2014-06-30']
    end_date = '2019-02-01'
    for interval, amazon_data, apple_date in zip(intervals, start_dates_amazon, start_dates_apple):
        apple_stock = Stock(ticker_symbol='aapl', company_name="apple", start_date=apple_date, end_date=end_date,
                            interval=interval)
        apple_stock.generate_metrics()
        apple_stock.save_data(semantic=False)
        amazon_stock = Stock(ticker_symbol='amzn', company_name="amazon", start_date=amazon_data, end_date=end_date,
                             interval=interval)
        amazon_stock.generate_metrics()
        amazon_stock.save_data(semantic=False)


if __name__ == '__main__':
    # generate_all_data()
    generate_all_models()
