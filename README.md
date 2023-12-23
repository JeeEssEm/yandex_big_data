## Охотники за аномалиями

* notebooks:
  * *LOF_RESULT/LOF algorithm.ipynb*
  * *prophet_result/TimeSeries_prophet.ipynb*
  * *isolation_forest.ipynb*
  * *explore_data_analysis.ipynb*
* algorithms:
  * *lof.py*
  * *isolation_forest.py*
  * *prophet_algo.py*

## Реализованные алгоритмы
* LOF
* Prophet
* Isolation forest

## Мини-документация
### AnomalyResponse
*(algorithms.core.response)*
#### Методы:
* *anomaly_points*: возвращает все аномальные точки
* *visualize*: отображает график и подсвечивает все аномальные точки

### AnomalyKit
*(algorithms.main)*
#### Методы:
* *get_confident_anomaly_points*: возвращает только те точки, которые все алгоритмы определили как аномальные
* *get_all_anomaly_points*: возвращает все найденные алгоритмами аномальные точки
* *get_algorithm_anomaly_points*: принимает на вход название алгоритма и возвращает найденные им аномальные точки
* *visualize*: принимает на вход название алгоритма и визуализирует обнаруженные им аномальные точки
