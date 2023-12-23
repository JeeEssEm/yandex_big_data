from sklearn.ensemble import IsolationForest
import utils.date
import utils.cinematic
import core.response
import plotly.express as px

import pandas as pd


def normalize_df(df: pd.DataFrame):
    df['date'] = df.apply(
        lambda row: utils.date.convert_to_datetime(row['pickup_datetime']),
        axis=1)

    start_time = df['date'].min()
    df['hour'] = df.apply(
        lambda row: (row['date'] - start_time).total_seconds() // 3600, axis=1)

    df['month'] = df.apply(lambda row: row['date'].strftime('%b %Y'), axis=1)
    df['distance'] = df.apply(lambda row: utils.cinematic.distance(
        row['pickup_longitude'],
        row['pickup_latitude'],
        row['dropoff_longitude'],
        row['dropoff_latitude'],
    ), axis=1)
    return df


def calc_anomalies(df, anomaly_inputs):
    model = IsolationForest(contamination=0.05, n_estimators=1000, max_samples=200)
    model.fit(df[anomaly_inputs])

    df['anomaly'] = model.predict(df[anomaly_inputs])
    return df


def get_xticks(df):
    return df.groupby(['month']).agg({'hour': 'max'}).reset_index()


def visualize(points_list):
    for df, col in points_list:
        months = get_xticks(df)
        fig = px.scatter(df, x='hour', y=col, color='anomaly',
                         color_continuous_scale=['orange', 'blue'])
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=months['hour'],
                ticktext=months['month'],
                title='month'
            ),
            yaxis_title=col,
        )
        fig.show()


def isolation_forest(df):
    df = normalize_df(df)

    trips_duration = df.groupby(['hour']).agg(
        {'trip_duration': 'mean'}).reset_index()
    trips_amount = df.groupby(['hour']).agg(
        trips_amount=('trip_duration', 'count')).reset_index()
    trips_distance = df.groupby(['hour']).agg(
        {'distance': 'mean'}).reset_index()

    trips_duration = calc_anomalies(trips_duration, ['trip_duration'])
    trips_amount = calc_anomalies(trips_amount, ['trips_amount'])
    trips_distance = calc_anomalies(trips_distance, ['distance'])

    t = df[['month', 'hour']].set_index('hour')
    trips_amount['month'] = trips_amount.apply(
        lambda row: t.loc[row['hour']].min(), axis=1)
    trips_distance['month'] = trips_distance.apply(
        lambda row: t.loc[row['hour']].min(), axis=1)
    trips_duration['month'] = trips_duration.apply(
        lambda row: t.loc[row['hour']].min(), axis=1)

    dur = trips_duration[trips_duration['anomaly'] == -1]
    amount = trips_amount[trips_amount['anomaly'] == -1]
    dist = trips_distance[trips_distance['anomaly'] == -1]

    anomalies = dist.merge(amount, on='hour').merge(dur, on='hour')
    anomaly_points = df[df['hour'].isin(anomalies['hour'])]

    all_points = [
        (trips_duration, 'trip_duration'),
        (trips_amount, 'trips_amount'),
        (trips_distance, 'distance'),
    ]

    return core.response.AnomalyResponse(
        dataset=df,
        all_points=all_points,
        anomaly_points=anomaly_points,
        visualize_func=visualize,
    )


if __name__ == '__main__':
    d = pd.read_csv('../notebooks/nyc_taxi_trip_duration.csv')
    resp = isolation_forest(d)
    print(resp.anomaly_points)
    resp.visualize()
