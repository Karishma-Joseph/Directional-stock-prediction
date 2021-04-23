import snscrape.modules.twitter as sntwitter
from datetime import date
import datetime
from textblob import TextBlob
import csv

class Twitter:
    ONE_MIN = "1m"
    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"
    THIRTY_MIN = "30m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"
    FIVE_DAY = "5d"
    SENTIMENT_SCORE = "Score"

    def getIntervalFromString(self, interval):
        switch = {
            "ONE_MIN": datetime.timeDelta(minutes=1),
            "FIVE_MIN": datetime.timeDelta(minutes=5),
            "FIFTEEN_MIN": datetime.timeDelta(minutes=15),
            "THIRTY_MIN": datetime.timeDelta(minutes=30),
            "ONE_HOUR": datetime.timeDelta(minutes=60),
            "ONE_DAY": datetime.timeDelta(days=1),
            "FIVE_DAY": datetime.timeDelta(days=5),
        }

        return switch.get(interval)

    def pullTwitterData(self, interval):
        #adjust fileName and path as needed
        fileName = "twitter_" + interval + "_18_19.csv"
        timeDelta = self.getIntervalFromString(interval)
        
        with open(fileName, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Date", "Score", "Frequency"])

            #this is end date of search, adjust as needed
            startTime = datetime.datetime(2020, 1, 1)
            total = 0
            totalCount = 0

            #change $AMZN to whatever string you're searching for, change dates accordingly
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper('$AMZN since:2018-01-01 until:2020-01-01').get_items()):
                if tweet.date.replace(tzinfo=None) < startTime - timeDelta:
                    startTime = startTime - timeDelta
                    time = startTime

                    averageSentiment = 0
                    if totalCount != 0:
                        averageSentiment = total/totalCount
                    writer.writerow([time, averageSentiment, totalCount])

                    while tweet.date.replace(tzinfo=None) < startTime - timeDelta:
                        startTime = startTime - timeDelta
                        writer.writerow([startTime, 0, 0])
                    total = 0
                    totalCount = 0
                totalCount+=1
                total += float(TextBlob(tweet.content).sentiment.polarity)

            while startTime > datetime.datetime(2018, 1, 1):
                startTime = startTime - timeDelta
                writer.writerow([startTime, 0, 0])
            file.close()
