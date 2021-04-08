import pandas as pd
import numpy as np
from selenium import webdriver
from io import StringIO
from datetime import datetime


class Metrics:
    SMA = '{}_sma_{}'
    EMA = '{}_ema_{}'
    INCREASE_DECREASE = '{}_increase_decrease'
    CROSSOVER_COUNT = '{}_crossover_count'
    DERIVATIVE = '{}_derivative'
    DISTANCE = '{}_{}_distance'
    CLASSIFICATION = '{}_classification'
    RSI = 'rsi'
    OBV = 'obv'
    MACD = 'macd'
    MACD_SIGNAL = 'macd_signal'
    MACD_DECISION = 'macd_decision'
    CLOSE_PRICE = 'Close'
    VOLUME = 'Volume'
    REVENUE = "revenue"
    PRICE_TO_BOOK = "price_to_book"
    EPS = "eps"
    PE = "pe_ratio"
    DEBT_EQUITY = "debt_to_equity"
    SPAN = 50
    ONE_MIN = "1m"
    FIVE_MIN = 5
    FIFTEEN_MIN = 15
    THIRTY_MIN = 30
    ONE_HOUR = 60
    FOUR_HOUR = 240
    ONE_DAY = 1440
    SENTIMENT_SCORE = "Score"
    NEWS_SENTIMENT_SCORE = "Sentiment score"
    FREQUENCY = "Frequency"


def increase_decrease(data, col):
    data['temp'] = data[col].shift(-1)
    data[Metrics.INCREASE_DECREASE.format(col)] = data.apply(lambda x: 1 if x['temp'] - x[col] > 0 else 0, axis=1)
    data = data.drop(columns=['temp'])
    return data


def column_derivative_metric(col, data):
    data[Metrics.DERIVATIVE.format(col)] = data[col].diff()
    return data


def distance_metric(data, col1, col2):
    data[Metrics.DISTANCE.format(col1, col2)] = (data[col1] - data[col2]) / data[col2]
    return data


def rsi_metric(data, window=14):
    # 1: Compute price movement each period
    # 2: Compute average gain/loss over last 14 days
    gain = pd.DataFrame(data[Metrics.CLOSE_PRICE].rolling(2).apply(lambda x: x.iloc[1] - x.iloc[0]))
    gain[gain < 0] = np.nan
    gain = gain.rolling(window=window, min_periods=1).mean()
    gain = gain.fillna(0)

    loss = pd.DataFrame(data[Metrics.CLOSE_PRICE].rolling(2).apply(lambda x: x.iloc[1] - x.iloc[0]))
    loss[loss > 0] = np.nan
    loss = loss.abs()
    loss = loss.rolling(window=window, center=True, min_periods=1).mean()
    loss = loss.fillna(0)
    # 3: Calculate RS and RSI
    relative_strength = gain / loss
    relative_strength_index = 100 - 100 / (1 + relative_strength)
    data[Metrics.RSI] = relative_strength_index
    return data


def macd_metric(data):
    # 12 period exponential moving average
    EMA_12_day = data[Metrics.CLOSE_PRICE].ewm(span=12, adjust=False).mean()
    # 26 period exponential moving average
    EMA_26_day = data[Metrics.CLOSE_PRICE].ewm(span=26, adjust=False).mean()
    data[Metrics.MACD] = EMA_26_day - EMA_12_day

    # 9 period exponential moving average
    data[Metrics.MACD_SIGNAL] = data[Metrics.MACD].ewm(span=9, adjust=False).mean()
    data[Metrics.MACD_DECISION] = data[Metrics.MACD] - data[Metrics.MACD_SIGNAL]
    data = data.drop(columns=[Metrics.MACD_SIGNAL, Metrics.MACD])
    return data


def on_balance_volume(data, span=12):
    # OBV = Previous OBV + Current Trading Volume
    data[Metrics.OBV] = np.where(data[Metrics.CLOSE_PRICE] > data[Metrics.CLOSE_PRICE].shift(1), data[Metrics.VOLUME],
                                 np.where(data[Metrics.CLOSE_PRICE] < data[Metrics.CLOSE_PRICE].shift(1),
                                          -data[Metrics.VOLUME], 0)).cumsum()
    # data = exponential_moving_average(data, metrics.OBV, span=span)
    # data = column_derivative_metric(metrics.EMA.format(metrics.OBV, span), data)
    return data


def simple_moving_average(data, col, window):
    data[Metrics.SMA.format(col, window)] = data[col].rolling(window=window).mean()
    return data


def exponential_moving_average(data, col, span):
    data[Metrics.EMA.format(col, span)] = data[col].ewm(span=span, adjust=False).mean()
    return data


def ema_trend_indicator(data, ema_span):
    # Required Column Names
    col_ema_name = Metrics.EMA.format(Metrics.CLOSE_PRICE, ema_span)
    col_derivative = Metrics.DERIVATIVE.format(col_ema_name)
    col_distance = Metrics.DISTANCE.format(col_ema_name, col_derivative)
    col_crossover_count = Metrics.CROSSOVER_COUNT.format(col_ema_name)
    col_ema_slope_classification = Metrics.CLASSIFICATION.format(col_ema_name)

    data[col_distance] = (data[Metrics.CLOSE_PRICE] - data[col_ema_name]) / data[col_ema_name]
    data[col_derivative] = data[col_ema_name].diff()
    data[col_crossover_count] = np.nan

    i = 1
    trend = 1
    for index, row in data.iterrows():
        if row[col_derivative] > 0:
            if trend == 1:
                i = i + 1
            else:
                trend = 1
                i = 1
        elif row[col_derivative] < 0:
            if trend == 0:
                row[col_crossover_count] = i
                i = i + 1
            else:
                trend = 0
                i = 1
        data.at[index, col_crossover_count] = i
    data['temp'] = data[col_derivative].shift(-1)
    data[col_ema_slope_classification] = data.apply(lambda x: 1 if x['temp'] > 0 else 0, axis=1)
    data = data.dropna()
    data = data.drop(columns=['temp', col_distance, col_derivative, col_ema_name])

    return data


def eps(data, ticker, company_name, period):
    browser = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")
    browser.get(
        f"https://www.macrotrends.net/stocks/charts/{ticker.upper()}/{company_name.lower()}/eps-earnings-per-share-diluted")
    tables = browser.find_element_by_id("style-1").find_elements_by_tag_name("table")
    for table in tables:
        title = table.text.split('\n', 1)
        table_data = StringIO(table.text.split('\n', 1)[-1])
        if period.lower() == "quarterly" and title[0].lower().find(period) != -1:
            eps_table = pd.read_csv(table_data, sep=' ', header=None, index_col=0)
            eps_table.index = pd.to_datetime(eps_table.index)
        if period.lower() == "annual" and title[0].lower().find(period) != -1:
            eps_table = pd.read_csv(table_data, sep=' ', header=None, index_col=0)
            eps_table.index = pd.to_datetime(eps_table.index)
    eps_table = pd.DataFrame(eps_table[eps_table.columns[0]].replace('[\$,]', '', regex=True).astype('float'))
    data = insert_data(data, eps_table, Metrics.EPS)
    browser.close()
    return data


def revenue(data, ticker, company_name, period):
    browser = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")
    browser.get(f"https://www.macrotrends.net/stocks/charts/{ticker.upper()}/{company_name.lower()}/revenue")
    tables = browser.find_element_by_id("style-1").find_elements_by_tag_name("table")
    for table in tables:
        title = table.text.split('\n', 1)
        table_data = StringIO(table.text.split('\n', 2)[-1])
        if period.lower() == "quarterly" and title[0].lower().find(period) != -1:
            revenue_table = pd.read_csv(table_data, sep=' ', header=None, index_col=0)
            revenue_table.index = pd.to_datetime(revenue_table.index)

        if period.lower() == "annual" and title[0].lower().find(period) != -1:
            revenue_table = pd.read_csv(table_data, sep=' ', header=None, index_col=0)
            revenue_table.index = pd.to_datetime(revenue_table.index)
    revenue_table = pd.DataFrame(revenue_table[revenue_table.columns[0]].replace('[\$,]', '', regex=True).astype('int'))
    data = insert_data(data, revenue_table, Metrics.REVENUE)
    browser.close()
    return data


def pe_ratio(data, ticker, company_name):
    browser = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")
    browser.get(f"https://www.macrotrends.net/stocks/charts/{ticker.upper()}/{company_name.lower()}/pe-ratio")
    table = browser.find_element_by_id("style-1").find_element_by_tag_name("table")

    title = table.text.split('\n', 1)
    table_data = StringIO(table.text.split('\n', 3)[-1])
    if title[0].lower().find("pe ratio") != -1:
        pe_ratio_table = pd.read_csv(table_data, sep=' ', header=None, index_col=0)
        pe_ratio_table.drop(pe_ratio_table.iloc[:, 1:3], inplace=True, axis=1)
        pe_ratio_table.index = pd.to_datetime(pe_ratio_table.index)
        data = insert_data(data, pe_ratio_table, Metrics.PE)
    browser.close()
    return data


def debt_to_equity_ratio(data, ticker, company_name):
    browser = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")
    browser.get(f"https://www.macrotrends.net/stocks/charts/{ticker.upper()}/{company_name.lower()}/debt-equity-ratio")
    table = browser.find_element_by_id("style-1").find_element_by_tag_name("table")

    title = table.text.split('\n', 1)
    table_data = StringIO(table.text.split('\n', 2)[-1])
    if title[0].lower().find("debt/equity") != -1:
        debt_to_equity_table = pd.read_csv(table_data, sep=' ', header=None, index_col=0)
        debt_to_equity_table.drop(debt_to_equity_table.iloc[:, 0:2], inplace=True, axis=1)
        debt_to_equity_table.index = pd.to_datetime(debt_to_equity_table.index)
        debt_to_equity_table.index = pd.to_datetime(debt_to_equity_table.index)
        data = insert_data(data, debt_to_equity_table, Metrics.DEBT_EQUITY)
    browser.close()
    return data


def price_to_book_ratio(data, ticker, company_name):
    browser = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")
    browser.get(f"https://www.macrotrends.net/stocks/charts/{ticker.upper()}/{company_name.lower()}/price-book")
    table = browser.find_element_by_id("style-1").find_element_by_tag_name("table")

    title = table.text.split('\n', 1)
    table_data = StringIO(table.text.split('\n', 3)[-1])
    if title[0].lower().find("price/book") != -1:
        price_to_book_table = pd.read_csv(table_data, sep=' ', header=None, index_col=0)
        price_to_book_table.drop(price_to_book_table.iloc[:, 0:2], inplace=True, axis=1)
        price_to_book_table.index = pd.to_datetime(price_to_book_table.index)
        data = insert_data(data, price_to_book_table, Metrics.PRICE_TO_BOOK)
    browser.close()
    return data


# Semantic News Features
def semantic_news_features(data, interval):
    # read data from /news_data/...
    path = "data/news_data/news_" + interval + ".csv"
    news_df = pd.read_csv(path)
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    news_df = news_df.set_index('Date')

    # insert data into dataframe
    data = insert_data(data, news_df, Metrics.NEWS_SENTIMENT_SCORE)

    return data


# Semantic News Features
def semantic_twitter_features(data, interval):
    # read data from /twitter_data/...
    path = "data/twitter_data/twitter_" + interval + ".csv"
    twitter_df = pd.read_csv(path)
    twitter_df['Date'] = pd.to_datetime(twitter_df['Date'])
    twitter_df = twitter_df.set_index('Date')

    # insert data into dataframe
    data = insert_data(data, twitter_df, Metrics.SENTIMENT_SCORE)
    data = insert_data(data, twitter_df, Metrics.FREQUENCY)

    return data



# Function inserts data according to date. If dates match or is between date range, data is inserted
# Fills in missing data in a forward fashion until next value occurs.
# Can be used with sentiment features as well
def insert_data(data, insert_data, column_name):
    # Make sure data is sorted the same direction
    data = data.sort_index(ascending=True)
    insert_data = insert_data.sort_index(ascending=True)
    new_data = insert_data.loc[data.index[0]:data.index[-1]]

    # Create new column with na values
    data[column_name] = np.nan

    # Runs in O(n). Looks for values between indices. Inserts these values in main data frame
    old_index = 0
    old_date = data.index.tolist()[0]
    for new_data_index, row in new_data.iterrows():
        for data_index, data_row in data.iloc[old_index:].iterrows():
            if new_data_index == data_index or (old_date < new_data_index < data_index):
                data.at[data_index, column_name] = row
                old_date = data_index
                old_index = data.index.get_loc(data_index)
                break
    # Forward fill any values that are not the same
    data = data.fillna(method='ffill')
    # Forward fill will leave the first values as na. Fill in these values with the value from the next smallest date index
    if insert_data.index[0] < data.index[0]:
        first_value = insert_data[insert_data.index < data.index[0]].sort_index(ascending=False).iloc[0, 0]
        data = data.fillna(first_value)
    data = data.dropna()

    return data
