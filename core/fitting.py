from core.config import dataframe, array


def exp_data(df: dataframe, columns: list) -> array:
    return df[columns].values


def obj_data(df: dataframe, col: str) -> array:
    return df[col].values


def predict(xs, coef, intercept):
    import numpy as np
    return np.dot(xs, coef) + intercept


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
