from config import dataframe, array


def exp_data(df: dataframe, columns: list, period: tuple) -> array:
    start, end = period if period else df.index[0], df.index[-1]
    return df.loc[start:end, columns].values


def obj_data(df: dataframe, col: str, period: tuple) -> array:
    import numpy as np
    start, end = period if period else df.index[0], df.index[-1]
    return np.array(df[col][start:end])


def least_squares(xs: array, y: array) -> tuple:
    import sklearn.linear_model as LinearModel
    reg = LinearModel.LinearRegression()
    reg.fit(xs, y)
    return reg.coef_, reg.intercept_


def ridge_regression(xs: array, y: array, alpha: float) -> tuple:
    import sklearn.linear_model as LinearModel
    reg = LinearModel.Ridge(alpha=alpha)
    reg.fit(xs, y)
    return reg.coef_, reg.intercept_


def lasso_regression(xs: array, y: array, alpha: float) -> tuple:
    import sklearn.linear_model as LinearModel
    reg = LinearModel.Lasso(alpha=alpha)
    reg.fit(xs, y)
    return reg.coef_, reg.intercept_
