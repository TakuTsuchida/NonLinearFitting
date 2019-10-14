import numpy as np
import pandas as pd

from core.fitting import exp_data, obj_data, least_squares, predict
from core.cross_validation import CrossValidation


def test_func(x_1, x_2):
    return 3 * x_1 - 2 * x_2 + np.random.randint(0, 100)


if __name__ == "__main__":
    # create data
    df = pd.DataFrame(data=np.random.randint(0, 100, (100, 3)),
                      columns=list('ABC'))
    df['C'] += test_func(df['A'], df['B'])
    train_df, test_df = df[0: 70], df[70:100]

    # regression
    coef, intercept = least_squares(
        xs=exp_data(df=train_df, columns=['A', 'B']),
        y=obj_data(df=train_df, col='C'))

    # cross validation
    y = obj_data(df=df, col='C')
    y_hat = predict(xs=exp_data(df=df, columns=['A', 'B']),
                    coef=coef,
                    intercept=intercept)
    CrossValidation.execute(y, y_hat)
