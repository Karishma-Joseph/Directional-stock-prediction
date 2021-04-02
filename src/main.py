from stock import Stock, Metrics
from models import *


if __name__ == '__main__':
    intervals = [Metrics.ONE_DAY]
    for interval in intervals:
        apple_stock = Stock(ticker_symbol='aapl', company_name="apple", start_date='2018-01-01', end_date='2018-2-31', interval=interval)
        apple_stock.generate_metrics()
        apple_stock.save_data()
        # amazon_stock = Stock(ticker_symbol='amzn', company_name="amazon", start_date='2018-01-01', end_date='2018-2-31')
        # amazon_stock.generate_metrics()
        # amazon_stock.save_data()