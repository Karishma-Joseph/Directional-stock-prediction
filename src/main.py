from stock import Stock, Metrics
from models import *
import os
import re

intervals = [Metrics.FIVE_MIN, Metrics.FIFTEEN_MIN, Metrics.THIRTY_MIN, Metrics.ONE_HOUR,
             Metrics.ONE_DAY]


# creates models for a given stock. pass in the feature variables and corresponding y column.
def generate_stock_models(x_col, y_cols, company, semantic):
    files = os.listdir("data/training_data/")
    for y_col in y_cols:
        for file in files:
            if company in file:
                if semantic and "semantic" not in file:
                    continue;
                data = pd.read_csv("data/training_data/{}".format(file))
                interval = re.search('(\d+)', file).group(0)
                model_type = generate_model_name(y_col, company, interval, semantic)
                generate_models(data=data, x_col=x_col, y_col=y_col, model_type=model_type)

# Generates the name for the model based on a number of parameters defined in this project
def generate_model_name(y_col_name, company, interval, semantic):
    if "classification" in y_col_name:
        model_type = ModelAttributes.EMA_CLASSIFICATION.format(company + "_" + str(interval) + intervals)
    else:
        model_type = ModelAttributes.INCREASE_DECREASE_CLASSIFICATION.format(
            company + "_" + str(interval) + intervals)
    if semantic:
        model_type += '_semantic'
    return model_type


##### NOTE: if you are going to generate data with semantic features, use save_data(semantic=True) (line 44, 48).
# If semantic = False, you will overwrite all of the data already generated by using an existing file name.
# Please do not do that!! It takes a long time to generate
def generate_all_data():
    # these are the start dates associated with the stock data I used. Will be different for semantic data.
    start_dates_amazon = ['2017-11-09', '2017-09-25', '2017-05-12', '2017-03-13', '2015-11-27', '2014-09-29']
    start_dates_apple = ['2017-07-12', '2016-12-22', '2016-08-23', '2016-03-16', '2014-06-30', '2014-06-30']
    end_date = '2019-02-01'
    for interval, amazon_date, apple_date in zip(intervals, start_dates_amazon, start_dates_apple):
        # apple_stock = Stock(ticker_symbol='aapl', company_name="apple", start_date=apple_date, end_date=end_date,
        #                     interval=interval)
        # # apple_stock.generate_metrics(semantic=True)
        # apple_stock.add_semantic_features()
        # apple_stock.save_data(semantic=True)
        amazon_stock = Stock(ticker_symbol='amzn', company_name="amazon", start_date=amazon_date, end_date=end_date,
                             interval=interval)
        amazon_stock.add_semantic_features()
        amazon_stock.save_data(semantic=True)


def demo():
    pass


if __name__ == '__main__':
    intervals = [Metrics.FIVE_MIN, Metrics.FIFTEEN_MIN, Metrics.THIRTY_MIN, Metrics.ONE_HOUR,
                 Metrics.ONE_DAY]
    companies = ['aapl', 'amzn']
    semantic = [True, False]

    col_ema_name = Metrics.EMA.format(Metrics.CLOSE_PRICE, Metrics.SPAN)
    col_derivative = Metrics.DERIVATIVE.format(col_ema_name)
    col_distance = Metrics.DISTANCE.format(col_ema_name, col_derivative)
    crossover_count = Metrics.CROSSOVER_COUNT.format(Metrics.EMA.format(Metrics.CLOSE_PRICE, Metrics.SPAN))

    x_col = [Metrics.RSI, Metrics.MACD_DECISION, Metrics.OBV, col_derivative, col_distance,
             crossover_count]
    x_col_semantic_aapl = ['Score', 'Frequency', 'Sentiment score']
    x_col_semantic_amzn = ['Score', 'Frequency']
    y_cols = [Metrics.INCREASE_DECREASE.format(Metrics.CLOSE_PRICE), 'Close_ema_50_classification']

    # Used to generate the data from the models
    # for is_semantic in semantic:
    #     if is_semantic:
    #         x_col_final = x_col + x_col_semantic_aapl
    #     else:
    #         x_col_final = x_col
    #     generate_stock_models(x_col_final, y_cols, 'aapl', is_semantic)

    for is_semantic in semantic:
        if is_semantic:
            x_col_final = x_col + x_col_semantic_amzn
        else:
            x_col_final = x_col
        generate_stock_models(x_col_final, y_cols, 'amzn', is_semantic)
    # generate_all_data()
