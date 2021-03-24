# from .stock import stock
import stock.stock_metrics as metrics
from stock import Stock

if __name__ == '__main__':
    stock = Stock(ticker_symbol='aapl', start_date='2020-01-01', end_date='2020-12-31')
    stock.generate_metrics()

# if __name__ == '__main__':
#     metrics.increase_decrease()
