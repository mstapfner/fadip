import numpy as np
import pandas as pd


# Based on https://towardsdatascience.com/time-series-of-price-anomaly-detection-13586cd5ff46
def get_distance_by_point(data, model):
    """
    Calculates the distance between the points in the data to its nearest centroid

    :param data: the data points
    :param model: the kmeans model
    """

    distance = pd.Series()
    for i in range(0, len(data)):
        x_a = np.array(data.loc[i])
        x_b = model.cluster_centers_[model.labels_[i] - 1]
        # distance.set_value(i, np.linalg.norm(x_a - x_b))
        distance.at[i] = np.linalg.norm(x_a - x_b)
    return distance
