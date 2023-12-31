import pandas as pd
from pyspark import SparkContext, SparkConf
import plotly.express as px
from prophet import Prophet
from datetime import datetime
from core.response import AnomalyResponse
from utils.cinematic import distance


def convert_to_datetime(str_datetime):
    return datetime(int(str_datetime[:4]), int(str_datetime[5:7]),
                    int(str_datetime[8:10]), int(str_datetime[11:13]))


def del_title(line):
    temp = line.split(',')
    if temp[0] != 'id':
        temp[2] = convert_to_datetime(temp[2])
        round_datetime = datetime(temp[2].year, temp[2].month, temp[2].day,
                                  temp[2].hour, 0, 0)
        return round_datetime, temp


def passenger_count(data):
    res = data.map(del_title) \
        .filter(lambda x: x != None) \
        .map(lambda x: (x[0], x[1][4])) \
        .reduceByKey(max)
    return res


def trip_count(data):
    result = data.map(del_title) \
        .filter(lambda x: x != None) \
        .map(lambda x: (x[0], 1)) \
        .reduceByKey(lambda a, b: a + b)
    return result


def dif_longitude(row):
    return row[0], (abs(float(row[1][7]) - float(row[1][5])), 1)


def dif_latitude(row):
    return row[0], (abs(float(row[1][8]) - float(row[1][6])), 1)


def metres(row):
    return row[0], (distance(*(map(float, row[1][5:9]))), 1)


def speed(row):
    return row[0], (distance(*(map(float, row[1][5:9]))) / int(row[1][10]), 1)


def duration(row):
    return row[0], (int(row[1][10]), 1)


def avg_time_series(data, func):
    result = data.map(del_title) \
        .filter(lambda x: x != None) \
        .map(func) \
        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
        .map(lambda x: (x[0], x[1][0] / x[1][1]))
    return result


def detect_anomalies(dataframe, isSeasonality=False):
    m = Prophet(daily_seasonality=isSeasonality, yearly_seasonality=False,
                weekly_seasonality=False,
                seasonality_mode='multiplicative',
                interval_width=0.99,
                changepoint_range=0.8)
    m.fit(dataframe)
    forecast = m.predict(dataframe)
    forecast['fact'] = dataframe['y'].astype(float)
    forecast['anomaly'] = (forecast['fact'] > forecast['yhat_upper']) | (
                forecast['fact'] < forecast['yhat_lower'])
    return forecast


def process(result, isSeasonality=False):
    temp = result.collect()
    temp = pd.DataFrame(temp, columns=['ds', 'y'])
    temp = temp.sort_values(by=['ds'])
    pred = detect_anomalies(temp, isSeasonality)
    fig = px.scatter(pred, x='ds', y='fact', color='anomaly',
                     color_continuous_scale=["orange", "blue", "red"])
    return pred[['ds', 'anomaly']], fig


prophet_metrics = [dif_longitude, dif_latitude, metres, speed, duration]
fig_list = []


def visual_prophet(point_lists=None):
    for x in fig_list:
        x.show()


class ProphetAnomaly:
    def __init__(self, df, dataset_name):
        self.dataset_name = dataset_name
        self.pyspark_data()
        self.df = df

    def result(self):
        global fig_list
        anomaly_df, fig_list = self.prophet_result()
        temp = anomaly_df.columns[1:]
        points_df = anomaly_df.copy()
        points_df['res_anomaly'] = False
        anomaly_return = pd.DataFrame(columns=anomaly_df.columns)
        normal_return = pd.DataFrame(columns=anomaly_df.columns)
        for index, row in anomaly_df.iterrows():
            temp = len(row)
            anomaly = False
            for x in range(1, temp):
                if row[x]:
                    anomaly = True
                    break
            if anomaly:
                anomaly_return = anomaly_return._append(row)
            else:
                normal_return = normal_return._append(row)
        return AnomalyResponse(self.df, anomaly_return, normal_return,
                               visual_prophet)

    def pyspark_data(self):
        conf = SparkConf().setAppName("test").setMaster("local")
        sc = SparkContext(conf=conf)
        self.pyspark_data = sc.textFile(self.dataset_name)

    def prophet_result(self):
        fig_list = []
        df_anomaly_list = []
        result = passenger_count(self.pyspark_data)
        pred, fig = process(result)
        pred = pred.rename(columns={'anomaly': 'anomaly_passenger_count'})
        fig_list.append(fig)
        df_anomaly_list.append(pred)
        result = trip_count(self.pyspark_data)
        pred, fig = process(result, True)
        pred = pred.rename(columns={'anomaly': 'anomaly_trip_count'})
        fig_list.append(fig)
        df_anomaly_list.append(pred)
        for x in prophet_metrics:
            result = avg_time_series(self.pyspark_data, x)
            pred, fig = process(result)
            pred = pred.rename(columns={'anomaly': 'anomaly_' + str(x)[10:]})
            fig_list.append(fig)
            df_anomaly_list.append(pred)
        for i in range(len(df_anomaly_list) - 1):
            df_anomaly_list[i + 1] = df_anomaly_list[i].merge(
                df_anomaly_list[i + 1], on='ds')
        return df_anomaly_list[-1], fig_list
