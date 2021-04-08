from stock import Stock, Metrics
from models import *
import os
import re

if __name__ == '__main__':
    # intervals = [Metrics.FIVE_MIN, Metrics.FIFTEEN_MIN, Metrics.THIRTY_MIN, Metrics.ONE_HOUR, Metrics.FOUR_HOUR , Metrics.ONE_DAY]
    # start_dates = ['2017-11-09', '2017-09-25', '2017-05-12', '2017-03-13', '2015-11-27', '2014-09-29']
    # for interval, date in zip(intervals, start_dates):
    #     apple_stock = Stock(ticker_symbol='aapl', company_name="apple", start_date='2016-12-22', end_date='2019-02-01', interval=Metrics.FIFTEEN_MIN)
    #     apple_stock.generate_metrics()
    #     apple_stock.save_data(semantic=False)
        # amazon_stock = Stock(ticker_symbol='amzn', company_name="amazon", start_date='2018-01-01', end_date='2018-2-31')
        # amazon_stock.generate_metrics()
        # amazon_stock.save_data()

    # ML model training
    companies = ['aapl', 'amzn']
    intervals = [Metrics.FIVE_MIN, Metrics.FIFTEEN_MIN, Metrics.THIRTY_MIN, Metrics.ONE_HOUR, Metrics.FOUR_HOUR,
                 Metrics.ONE_DAY]
    x_col = [Metrics.RSI, Metrics.MACD_DECISION, Metrics.OBV,
             Metrics.CROSSOVER_COUNT.format(Metrics.EMA.format(Metrics.CLOSE_PRICE, Metrics.SPAN)), Metrics.EPS,
             Metrics.REVENUE,
             Metrics.PE, Metrics.DEBT_EQUITY, Metrics.PRICE_TO_BOOK]
    # y_col = [Metrics.INCREASE_DECREASE.format(Metrics.CLOSE_PRICE)]
    y_col = ['Close_ema_50_classification']
    for company in companies:
        for file in os.listdir("data/training_data/"):
            if company in file:
                data = pd.read_csv("data/training_data/{}".format(file))
                interval = re.search('(\d+)', file).group(0)
                model_type = ModelAttributes.EMA_CLASSIFICATION.format(company + "_" + str(interval))
                name = ModelAttributes.LOGISTIC_REGRESSION.format(model_type)
                generate_models(data=data, x_col=x_col, y_col=y_col, model_type=model_type)
