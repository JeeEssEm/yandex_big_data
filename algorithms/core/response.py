class AnomalyResponse:
    def __init__(self, dataset, all_points, anomaly_points, visualize_func=None):
        self._dataset = dataset
        self._all_points = all_points
        self._anomaly_points = anomaly_points
        self._visualize_func = visualize_func

    @property
    def df(self):
        """Возвращает исходный датасет"""
        return self._dataset

    @property
    def anomaly_points(self):
        """Возвращает все аномальные точки"""
        return self._anomaly_points

    def visualize(self):
        """Отображает график и подсвечивает все аномальные точки"""
        if self._visualize_func is not None:
            self._visualize_func(self._all_points, self._anomaly_points)

    def __repr__(self):
        return f'<Response> anomaly points count: {len(self._anomaly_points)}'
