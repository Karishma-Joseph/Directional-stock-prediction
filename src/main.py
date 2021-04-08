from stock import Stock, Metrics
from models import *


if __name__ == '__main__':
    intervals = [Metrics.FIVE_MIN, Metrics.FIFTEEN_MIN, Metrics.THIRTY_MIN, Metrics.ONE_HOUR, Metrics.FOUR_HOUR , Metrics.ONE_DAY]
    start_dates = ['2017-11-09', '2017-09-25', '2017-05-12', '2017-03-13', '2015-11-27', '2014-09-29']
    for interval, date in zip(intervals, start_dates):
        apple_stock = Stock(ticker_symbol='amzn', company_name="amazon", start_date=date, end_date='2019-02-01', interval=interval)
        apple_stock.generate_metrics()
        apple_stock.save_data(semantic=False)
        # amazon_stock = Stock(ticker_symbol='amzn', company_name="amazon", start_date='2018-01-01', end_date='2018-2-31')
        # amazon_stock.generate_metrics()
        # amazon_stock.save_data()