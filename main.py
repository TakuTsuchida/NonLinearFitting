import numpy as np
import pandas as pd

from core.fitting import exp_data, obj_data, least_squares, predict
from core.cross_validation import CrossValidation
from core.k_mean import k_mean_df


def test_func(x_1, x_2):
    return 3 * x_1 - 2 * x_2 + np.random.randint(0, 100)


if __name__ == "__main__":
    # create data
    df = pd.DataFrame(data=np.random.randint(0, 100, (100, 3)),
                      columns=list('ABC'))
    df['C'] += test_func(df['A'], df['B'])

    # regression
    coef, intercept = least_squares(
        xs=exp_data(df=df, columns=['A', 'B']),
        y=obj_data(df=df, col='C'))

    # cross validation
    k_mean_df = k_mean_df(df, 15)
    y = obj_data(df=k_mean_df, col='C')
    y_hat = predict(xs=exp_data(df=k_mean_df, columns=['A', 'B']),
                    coef=coef,
                    intercept=intercept)
    CrossValidation.execute(y, y_hat)

    # calc BIAS and RMSE
