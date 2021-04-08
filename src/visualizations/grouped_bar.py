import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import csv
import glob

companies = ['aapl', 'amzn']
time_intervals = ["5", "15", "30", "60", "240", "1440"]

for company in companies:
    for time in time_intervals:
        res = []
        for method in ['ema', 'increase_decrease']:
            accuracy_list = []
            for model in ['decision_tree', 'logistic', 'random']:
                for fpath in glob.glob('../models/model_metrics/'+model+'*'+method+'*'+company+'_'+time+'*'):
                    with open(fpath, "r") as f:
                        line = f.readline()
                        while line:
                            line = f.readline()
                            line_split = line.split(",")
                            if len(line) >= 3:
                                accuracy_list.append(round(float(line_split[3]), 4) * 100)
            res.append(accuracy_list)

        fig = plt.figure(figsize=(10, 5))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)

        labels = ['DT', 'LR', 'Random Forest']
        ema_accuracy = res[0]
        increase_decrease_accuracy = res[1]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        # fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, ema_accuracy, width, label='EMA', color="#B19CD9")
        rects2 = ax.bar(x + width/2, increase_decrease_accuracy, width, label='Increase/Decrease', color="#86CBED")

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Accuracy (%) ')
        ax.set_title('Comparision of models')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()

        plt.savefig("without_sentiment_"+company+"_"+time+".jpg")
