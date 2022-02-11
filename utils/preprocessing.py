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
            gmean_ = gmean(transformed.values[np.where(transformed != 0)])
            for col in transformed.columns:
                col_tr = np.log(transformed[col][transformed[col]>0] /\
                 gmean_)
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


if __name__ == "__main__":

    # TODO create unit test
    X = pd.DataFrame(np.array([[1,0,3,6], [4,5,6,7], [2,5.5,6,8.2]]))

    expected_1_None = np.array([[-0.96345725,  0.        ,  0.13515504,  0.82830222],
                            [-0.29705611, -0.07391256,  0.108409  ,  0.26255968],
                            [-0.88030004,  0.13130087,  0.21831225,  0.53068693]])

    expected_0_None = np.array([[-0.69314718,  0.        , -0.46209812, -0.15550845],
                                [ 0.69314718, -0.04765509,  0.23104906, -0.00135778],
                                [ 0.        ,  0.04765509,  0.23104906,  0.15686623]])
    
    expected_True = np.array([[-1.44705114,  0.        , -0.34843885,  0.34470833],
                              [-0.06075678,  0.16238677,  0.34470833,  0.49885901],
                              [-0.75390396,  0.25769695,  0.34470833,  0.65708301]])

    transformer = RCLRTransformer()
    res_1_None = transformer.fit_transform(X)
    assert np.allclose(res_1_None.values, expected_1_None)

    transformer = RCLRTransformer(axis=0)
    res_0_None = transformer.fit_transform(X)
    assert np.allclose(res_0_None.values, expected_0_None)
   
    transformer = RCLRTransformer(global_gmean=True)
    res_0_None = transformer.fit_transform(X)
    assert np.allclose(res_0_None.values, expected_True)

    transformer = RCLRTransformer(axis=0, global_gmean=True)
    res_0_None = transformer.fit_transform(X)
    assert np.allclose(res_0_None.values, expected_True)