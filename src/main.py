# from .stock import stock
import stock.stock_metrics as metrics
from stock import Stock

if __name__ == '__main__':
    apple_stock = Stock(ticker_symbol='aapl', company_name="apple", start_date='2018-01-01', end_date='2020-12-31')
    apple_stock.generate_metrics()
    amazon_stock = Stock(ticker_symbol='amzn', company_name="amazon", start_date='2018-01-01', end_date='2020-12-31')
    amazon_stock.generate_metrics()
