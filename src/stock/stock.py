import yfinance as yf

from .stock_metrics import *
from datetime import datetime, timedelta


class Stock:

    def __init__(self, ticker_symbol, company_name, start_date, end_date, interval):
        self.START_DATE = start_date
        self.END_DATE = end_date
        self.ticker_symbol = ticker_symbol
        self.company_name = company_name
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.get_stock_data(company_name, datetime.strptime(start_date, '%Y-%m-%d'),
                                        datetime.strptime(end_date, '%Y-%m-%d'), interval)

    def get_stock_data(self, company_name, start_date, end_date, interval):
        columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        file = "data/stock_data/{}.csv".format(str.upper(company_name) + str(interval))
        data = pd.read_csv(file, names=columns, parse_dates=[['Date', 'Time']]).set_index(['Date_Time'], drop=True)
        data = data.loc[start_date:end_date]
        return data

    def generate_metrics(self):
        # Technical Indicators
        self.data = rsi_metric(data=self.data)
        self.data = macd_metric(data=self.data)
        self.data = on_balance_volume(data=self.data)
        self.data = exponential_moving_average(data=self.data, col=Metrics.CLOSE_PRICE, span=Metrics.SPAN)
        self.data = ema_trend_indicator(data=self.data, ema_span=Metrics.SPAN)
        self.data = increase_decrease(data=self.data, col=Metrics.CLOSE_PRICE)

        # Fundamental Indicators
        self.data = revenue(data=self.data,  ticker=self.ticker_symbol, company_name=self.company_name, period="quarterly")
        self.data = eps(data=self.data,  ticker=self.ticker_symbol, company_name=self.company_name, period="quarterly")
        self.data = pe_ratio(data=self.data, ticker=self.ticker_symbol, company_name=self.company_name)
        self.data = debt_to_equity_ratio(data=self.data, ticker=self.ticker_symbol, company_name=self.company_name)
        self.data = price_to_book_ratio(data=self.data, ticker=self.ticker_symbol, company_name=self.company_name)

        # Semantic News Features
        # Match the times and insert semantic news features
        # self.data = semantic_news_features(data=self.data)
        #
        # Semantic Twitter Features
        # self.data = semantic_twitter_features(data=self.data)
        return

    def save_data(self, semantic):
        if semantic:
            file_name = "data/training_data/{}_{}_{}_{}_{}.csv".format(self.ticker_symbol, self.interval, self.start_date, self.end_date, "semantic")
        else:
            file_name = "data/training_data/{}_{}_{}_{}.csv".format(self.ticker_symbol, self.interval, self.start_date, self.end_date)
        self.data.to_csv(file_name)






    ## Useless function. Unfortunately, yahoo finance would not work for intraday trading
    def get_yf_stock_data(self, ticker_symbol, start_date, end_date, interval):
        # yf api only allows intraday stock info for < 60 days. So multiple api calls will be necessary
        current_date = start_date
        temp = (end_date - current_date).days
        if (end_date - current_date).days <= 60:
            data = pd.DataFrame(
                yf.download(
                    ticker_symbol,
                    start=current_date.date().strftime('%Y-%m-%d'),
                    END_DATE=(current_date + timedelta(days=60)).date().strftime('%Y-%m-%d'),
                    interval=interval,
                    progress=False))
            return data
        else:
            temp_date = (current_date + timedelta(days=2)).date().strftime('%Y-%m-%d')
            data = pd.DataFrame(
                yf.download(
                    ticker_symbol,
                    start=current_date,
                    # END_DATE=(current_date + timedelta(days=2)).date().strftime('%Y-%m-%d'),
                    interval=interval,
                    progress=False))
            current_date = current_date + 60
            while current_date - end_date > 60:
                temp = pd.DataFrame(
                    yf.download(
                        ticker_symbol,
                        start=current_date,
                        END_DATE=(current_date + timedelta(days=60)).date().strftime('%Y-%m-%d'),
                        interval=interval,
                        progress=False))
                data.append(temp)
                current_date = current_date + timedelta(days=60)
            return data

