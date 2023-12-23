from isolation_forest import isolation_forest
from lof import lof
import pandas as pd
from prophet_algo import ProphetAnomaly
import enum


class Algorithm(enum.Enum):
    IsolationForest = 'isolation forest'
    Prophet = 'prophet'
    LOF = 'lof'


def get_algo_name(name):
    if name not in [item.value for item in Algorithm]:
        raise Exception('Такого алгоритма не существует!')
    return Algorithm(name)


class AnomalyKit:
    def __init__(self, dataset_path: str):
        """:param str dataset_path: путь к датасету"""
        self._responses = {
            Algorithm.Prophet: None,
            Algorithm.LOF: None,
            Algorithm.IsolationForest: None,
        }
        self.df_path = dataset_path
        self._df = pd.read_csv(dataset_path)

    def get_confident_anomaly_points(self):
        """ Получить только те точки, которые все алгоритмы посчитали
        аномальными """
        lof_alg = self.get_algorithm_anomaly_points(Algorithm.LOF)
        prophet_alg = self.get_algorithm_anomaly_points(Algorithm.Prophet)
        isolation_forest_alg = self.get_algorithm_anomaly_points(
            Algorithm.IsolationForest
        )
        rs = lof_alg.merge(isolation_forest_alg, on='id').merge(
            prophet_alg, on='id'
        )
        return self._df[self._df['id'].isin(rs['id'])]

    def get_all_anomaly_points(self):
        """ Получить все точки, которые алгоритмы посчитали аномальными """
        lof_alg = self.get_algorithm_anomaly_points(Algorithm.LOF)
        prophet_alg = self.get_algorithm_anomaly_points(Algorithm.Prophet)
        isolation_forest_alg = self.get_algorithm_anomaly_points(
            Algorithm.IsolationForest
        )

        return pd.concat(
            [lof_alg, isolation_forest_alg, prophet_alg]
        ).drop_duplicates()

    def get_algorithm_anomaly_points(self, algo_name):
        """:arg algo_name: название одного из алгоритмов
         ('prophet', 'lof', 'isolation forest') для получения аномальных точек
        """
        if isinstance(algo_name, str):
            algo_name = get_algo_name(algo_name)

        if self._responses.get(algo_name) is None:
            self._calculate_anomaly_points([algo_name])
        return self._responses[algo_name].anomaly_points

    def _calculate_anomaly_points(self, algos):
        algorithms = {
            Algorithm.Prophet: lambda d: ProphetAnomaly(d, self.df_path).result(),
            Algorithm.LOF: lof,
            Algorithm.IsolationForest: isolation_forest,
        }

        for algo in algos:
            if isinstance(self._responses.get(algo, False), bool):
                raise Exception('Такого алгоритма не существует!')
            elif self._responses.get(algo, False) is None:
                self._responses[algo] = algorithms[algo](self._df)

    def visualize(self, algo_name):
        """:arg str algo_name: название одного из алгоритмов
        ('prophet', 'lof', 'isolation forest') для визуализации аномальных точек
        """
        algo_name = get_algo_name(algo_name)
        self._calculate_anomaly_points([algo_name])
        self._responses[algo_name].visualize()
