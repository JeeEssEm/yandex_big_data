import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkContext, SparkConf
import plotly.express as px
from prophet import Prophet
import math
from datetime import datetime, timedelta

def convert_to_datetime(str_datetime):
  return datetime(int(str_datetime[:4]), int(str_datetime[5:7]), int(str_datetime[8:10]), int(str_datetime[11:13]), int(str_datetime[14:16]), int(str_datetime[17:19]))

def del_title(line):
  temp = line.split(',')
  if temp[0] != 'id':
    temp[2] = convert_to_datetime(temp[2])
    round_datetime = datetime(temp[2].year, temp[2].month, temp[2].day, temp[2].hour, 0, 0)
    return round_datetime, temp

def passenger_count(data):
  res = data.map(del_title)\
  .filter(lambda x: x!= None)\
  .map(lambda x: (x[0], x[1][4]))\
  .reduceByKey(max)
  return res

def trip_count(data):
  result = data.map(del_title)\
  .filter(lambda x: x != None)\
  .map(lambda x: (x[0], 1))\
  .reduceByKey(lambda a, b: a + b)
  return result

def dif_longitude(row):
  return row[0], (abs(float(row[1][7]) - float(row[1][5])), 1)

def dif_latitude(row):
  return row[0], (abs(float(row[1][8]) - float(row[1][6])), 1)

def distance(test):
  longitude1, latitude1, longitude2, latitude2 = map(float, test)
  degree_kilometres = 111.2
  latitude_dif = degree_kilometres * abs(latitude2 - latitude1)
  longitude_dif = abs(longitude1 - longitude2)
  AD = degree_kilometres * math.cos(math.radians(latitude1)) * longitude_dif
  BC = degree_kilometres * math.cos(math.radians(latitude2)) * longitude_dif
  temp = (AD - BC) / 2
  H = (latitude_dif ** 2 - temp ** 2) ** 0.5
  return ((max(AD, BC) - temp) ** 2 + H ** 2) ** 0.5 * 1000

def metres(row):
  return row[0], distance(row[1][5:9])

def speed(row):
  return row[0], distance(row[1][5:9]) / float(row[10])

def duration(row):
  return row[0], float(row[10])

def avg_time_series(data, func):
  result = data.map(del_title)\
  .filter(lambda x: x!=None)\
  .map(func)\
  .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))\
  .map(lambda x: (x[0], x[1][0] / x[1][1]))
  return result

def detect_anomalies(forecast, isSeasonality=False):
    m = m = Prophet(daily_seasonality = isSeasonality, yearly_seasonality = False, weekly_seasonality = False,
                seasonality_mode = 'multiplicative',
                interval_width = 0.99,
                changepoint_range = 0.8)
    m.fit(forecast)
    forecast = m.predict(forecast)
    if isSeasonality:
      forecast['anomaly'] = forecast['yhat'] > forecast['trend_upper'] + forecast['additive_terms_upper']
      forecast[forecast['anomaly'] == 0]['anomaly'] = -1 * forecast['yhat'] < forecast['trend_lower'] + forecast['additive_terms_lower']
    else:
      forecast['anomaly'] = forecast['yhat'] > forecast['trend_upper']
      forecast[forecast['anomaly'] == 0]['anomaly'] = -1 * forecast['yhat'] < forecast['trend_lower']
    forecast['fact'] = forecast['yhat']
    return forecast

def process(result, isSeasonality=False):
  temp = result.collect()
  temp = pd.DataFrame(temp, columns=['ds', 'y'])
  temp = temp.sort_values(by=['ds'])
  pred = detect_anomalies(temp, isSeasonality)
  fig = px.scatter(pred, x='ds', y='fact', color='anomaly', color_continuous_scale=["orange", "blue", "red"])
  return pred[['ds', 'anomaly']], fig

prophet_metrics = [dif_longitude, dif_latitude, metres, speed, duration]
class AnomalyKit:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.pyspark_data()
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
      pred = pred.rename(columns={'anomaly' : 'anomaly_trip_count'})
      fig_list.append(fig)
      df_anomaly_list.append(pred)
      for x in prophet_metrics:
        result = avg_time_series(self.pyspark_data, x)
        try:
          pred, fig = process(result)
        except:
          print(x)
        pred = pred.rename(columns={'anomaly' : 'anomaly_'+str(x)})
        fig_list.append(fig)
        df_anomaly_list.append(pred)
      for i in range(len(df_anomaly_list) - 1):
        df_anomaly_list[i + 1] = df_anomaly_list[i].merge(df_anomaly_list[i + 1], on='ds')
      return df_anomaly_list[-1], fig_list
