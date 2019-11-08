import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_obs, y_pred):
    np.sqrt(mean_squared_error(y_obs, y_pred))
