from stocks.stock import stock


if __name__ == '__main__':
    stock = stock(ticker_symbol='aapl', start_date='2020-01-01', end_date='2020-12-31')
    stock.generate_metrics()
