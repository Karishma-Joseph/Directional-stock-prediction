import yfinance as yf
from .stock_metrics import *


class Stock:
    LINEAR_REGRESSION_EMA_SPAN = 50

    def __init__(self, ticker_symbol, start_date, end_date):
        self.START_DATE = start_date
        self.END_DATE = end_date
        self.ticker_symbol = ticker_symbol
        self.data = yf.download(ticker_symbol,
                                start=self.START_DATE,
                                END_DATE=self.END_DATE,
                                progress=False)

    def generate_metrics(self):
        self.data = rsi_metric(data=self.data)
        self.data = macd_metric(data=self.data)
        self.data = on_balance_volume(data=self.data)
        self.data = exponential_moving_average(data=self.data, col=metrics.CLOSE_PRICE,
                                               span=self.LINEAR_REGRESSION_EMA_SPAN)
        self.data = ema_trend_indicator(data=self.data, ema_span=self.LINEAR_REGRESSION_EMA_SPAN)
        self.data = increase_decrease(data=self.data, col=metrics.CLOSE_PRICE)
        return

    def get_stock_data(self):
        return self.data
