from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import matplotlib.pyplot as plt
import core.response
import utils.cinematic


def algorithm(file, factors, n, n_neighbors=200):
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    file[f'is_anomaly_step_{n + 1}'] = clf.fit_predict(file[factors])
    return file


def gluing(anom):
    res = anom[0]
    for i in range(1, len(anom)):
        res = pd.merge(res, anom[i], how="left", on=res.columns.values.tolist())
    return res


def process(file, factors, iterations=3, neighbors_nums=None):
    if neighbors_nums is None:
        neighbors_nums = [300, 500, 20]
    anomalies = []
    for i in range(iterations):
        n_file = algorithm(file, factors, i, n_neighbors=neighbors_nums[i])
        anomalies.append(n_file)
        file = n_file[n_file[f'is_anomaly_step_{i + 1}'] == -1]
    return anomalies


def visualize(all_points):
    plt.scatter(all_points['trip_duration'], all_points['distance'],
                c=all_points['is_anomaly_step_3'])
    plt.xlabel('Время поездки (trip_duration)')
    plt.ylabel('Дистанция поездки (distance)')
    plt.show()


def lof(df):
    factors = ['passenger_count', 'trip_duration', 'distance']
    df['distance'] = df.apply(lambda row: utils.cinematic.distance(
        row['pickup_longitude'],
        row['pickup_latitude'],
        row['dropoff_longitude'],
        row['dropoff_latitude'],
    ), axis=1)

    all_points = process(df, factors)

    glued_points = gluing(all_points)
    anomaly_points = glued_points[glued_points['is_anomaly_step_3'] == -1]

    return core.response.AnomalyResponse(
        dataset=df,
        all_points=glued_points,
        anomaly_points=anomaly_points,
        visualize_func=visualize
    )
