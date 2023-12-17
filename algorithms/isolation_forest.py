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

    df['day'] = df.apply(lambda row: int(row['date'].strftime('%j')), axis=1)
    df['month'] = df.apply(lambda row: row['date'].strftime('%b %Y'), axis=1)
    df['distance'] = df.apply(lambda row: utils.cinematic.distance(
        row['pickup_longitude'],
        row['pickup_latitude'],
        row['dropoff_longitude'],
        row['dropoff_latitude'],
    ), axis=1)
    return df


def calc_anomalies(df, anomaly_inputs):
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(df[anomaly_inputs])

    df['anomaly'] = model.predict(df[anomaly_inputs])
    return df


def get_xticks(df):
    return df.groupby(['month']).agg(day=('day', 'max')).reset_index()


def visualize(points_list):
    for df, col in points_list:
        months = get_xticks(df)
        fig = px.scatter(df, x='day', y=col, color='anomaly',
                         color_continuous_scale=['orange', 'blue'])
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=months['day'],
                ticktext=months['month'],
            )
        )
        fig.show()


def isolation_forest(df):
    df = normalize_df(df)

    amount_trips_df = df.groupby(['day', 'month']).agg(
        amount_of_trips=('trip_duration', 'count')
    ).reset_index()
    trip_durations_df = df.groupby(['day', 'month']).agg(
        trips_time=('trip_duration', 'sum')
    ).reset_index()
    trip_distance_df = df.groupby(['day', 'month']).agg(
        distance=('distance', 'sum')
    ).reset_index()

    amount_trips_df = calc_anomalies(amount_trips_df, ['day', 'amount_of_trips'])
    trip_durations_df = calc_anomalies(trip_durations_df, ['day', 'trips_time'])
    trip_distance_df = calc_anomalies(trip_distance_df, ['day', 'distance'])

    anomaly_points = (
            df[df['day'].isin(amount_trips_df[amount_trips_df['anomaly'] == -1]['day'])],
            df[df['day'].isin(trip_distance_df[trip_distance_df['anomaly'] == -1]['day'])],
            df[df['day'].isin(trip_durations_df[trip_durations_df['anomaly'] == -1]['day'])]
    )

    anomaly_points = pd.concat(anomaly_points)
    all_points = [
        (amount_trips_df, 'amount_of_trips'),
        (trip_distance_df, 'distance'),
        (trip_durations_df, 'trips_time')
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
