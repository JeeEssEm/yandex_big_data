import datetime as dt


def convert_to_datetime(date):
    return dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
