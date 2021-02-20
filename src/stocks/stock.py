import yfinance as yf


class stock:
    START_DATE = '2005-01-01'
    END_DATE = '2020-12-31'

    def __init__(self, ticker_symbol, period):
        self.ticker_symbol = ticker_symbol
        self.data = yf.download(ticker_symbol,
                                start=self.START_DATE,
                                END_DATE=self.END_DATE,
                                progress=False)

    def get_stock_data(self):
        return self.data
