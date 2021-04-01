import yfinance as yf
from .stock_metrics import *


class Stock:
    SPAN = 50

    def __init__(self, ticker_symbol, company_name, start_date, end_date):
        self.START_DATE = start_date
        self.END_DATE = end_date
        self.ticker_symbol = ticker_symbol
        self.company_name = company_name
        self.data = yf.download(ticker_symbol,
                                start=self.START_DATE,
                                END_DATE=self.END_DATE,
                                progress=False)

    def generate_metrics(self):
        # Technical Indicators
        self.data = rsi_metric(data=self.data)
        self.data = macd_metric(data=self.data)
        self.data = on_balance_volume(data=self.data)
        self.data = exponential_moving_average(data=self.data, col=metrics.CLOSE_PRICE, span=self.SPAN)
        self.data = ema_trend_indicator(data=self.data, ema_span=self.SPAN)
        self.data = increase_decrease(data=self.data, col=metrics.CLOSE_PRICE)

        # Fundamental Indicators
        self.data = revenue(data=self.data,  ticker=self.ticker_symbol, company_name=self.company_name, period="quarterly")
        self.data = eps(data=self.data,  ticker=self.ticker_symbol, company_name=self.company_name, period="quarterly")
        self.data = pe_ratio(data=self.data, ticker=self.ticker_symbol, company_name=self.company_name)
        self.data = debt_to_equity_ratio(data=self.data, ticker=self.ticker_symbol, company_name=self.company_name)
        self.data = price_to_book_ratio(data=self.data, ticker=self.ticker_symbol, company_name=self.company_name)
        return

