import numpy as np
import pandas as pd
from scipy.stats import gmean


class RCLRTransformer:
    """
    Transform pandas.DataFrame using RCLR.

    Remarks:
    - By defult, dataframe is transformed row-wise (axis=1),
      which preserves mean=0 across time points.
      This is different from the Scikit-Learn scalers.
    - If global_mean=True, elements are transformed
      using global geometric mean. Note that in such case
      transformed data will not be zero-centered in any direction.

    Return: transformed dataframe (copy)

    TODO:
     - the method requires optimization (works too slow)
    """

    def __init__(self, axis=1, global_gmean=None):
        self.axis = axis
        self.global_gmean = global_gmean

    def fit_transform(self, X):
        transformed = X.copy()
        if self.axis == 1:
            transformed = transformed.T
        if self.global_gmean:
            for col in transformed.columns:
                col_tr = np.log(transformed[col][transformed[col]>0] /\
                 self.global_gmean)
                transformed.loc[col_tr.index, col] = col_tr
        else:
            for col in transformed.columns:
                col_tr = np.log(transformed[col][transformed[col]>0] /\
                                gmean(transformed[col][transformed[col]>0]))
                transformed.loc[col_tr.index, col] = col_tr
        if self.axis == 1:
            return transformed.T
        else:
            return transformed
        return transformed

    def fit(self, X):
        # TODO
        return self

    def transform(self, X):
        # TODO
        return X

    def inverse_transform(self, X):
        # TODO
        return X


