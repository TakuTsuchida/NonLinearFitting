import numpy as np
import pandas as pd

from core.config import array, dataframe


def euclid_distance(x: array, y: array) -> float:
    return np.sqrt(sum(np.power(x-y, 2)))


def intermediate(x: array, y: array) -> array:
    return (x+y)/2


def k_mean_df(df: dataframe, k: int) -> dataframe:
    def extract_k_distance(x: array) -> dataframe:
        ed_data = [euclid_distance(x, df.loc[j].values) for j in df.index]
        ed_df = pd.DataFrame(
            ed_data,
            index=df.index,
            columns=['euclid_distance'])
        ed_df = pd.concat([df, ed_df], axis=1)
        k_mean = ed_df.sort_values('euclid_distance')[0:k]
        del k_mean['euclid_distance']
        return k_mean

    def k_mean_matrix(k_mean: dataframe) -> list:
        start = k_mean.index[0]
        inter_data = map(lambda i: intermediate(
            k_mean.loc[start].values,
            k_mean.loc[i].values), k_mean.index)
        return list(inter_data)

    matrixs = []
    for i in df.index:
        k_mean = extract_k_distance(df.loc[i].values)
        k_mean_data = k_mean_matrix(k_mean)
        matrixs.extend(k_mean_data)
    return pd.DataFrame(matrixs, columns=df.columns)
