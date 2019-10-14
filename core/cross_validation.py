import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CrossValidation:
    @classmethod
    def execute(cls, y, y_hat):
        error = cls.error(y, y_hat)
        df = cls.df_from_data(y, y_hat, error)
        cls.view(df)

    @staticmethod
    def error(y, y_hat):
        return np.abs(y-y_hat)

    @staticmethod
    def df_from_data(y, y_hat, error):
        data = np.array([y, y_hat, error]).T
        return pd.DataFrame(data=data,
                            columns=['actual', 'predict', 'error'])

    @staticmethod
    def view(df):
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.plot(df.index, df.actual, label='actual')
        ax.plot(df.index, df.predict, label='predict')
        ax.plot(df.index, df.error, label='error')
        ax.legend()
        ax.set_title('cross validation')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
        # plt.savefig('data/graph.png')
