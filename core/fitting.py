import pandas as pd
import numpy as np


def exp_data(df, columns: list, period: tuple):
    start, end = period
    return df.loc[start:end, columns].values


def obj_data(df, col: str, period: tuple):
    import numpy as np
    start, end = period
    return np.array(df[col][start:end])


def least_squares(xs, y):
    import sklearn.linear_model as LinearModel
    reg = LinearModel.LinearRegression()
    reg.fit(xs, y)
    return reg.coef_, reg.intercept_


if __name__ == "__main__":
    data = np.random.randint(0, 100, (100, 5))
    df = pd.DataFrame(data, columns=list("ABCDE"))
    period = (df.index[0], df.index[-1])
    coef, const = least_squares(xs=exp_data(df, ['A', 'B'], period),
                                y=exp_data(df, 'D', period))
    print(coef, const)
