import numpy as np
import pandas as pd
import numpy.ma as ma

from scipy.stats import gmean


class RCLRTransformer:
    """Transform features using Robust Centered Log-Ratio 
    transformation (RCLR).
    
    - By defult, a table is transformed column-wise (axis=1),
      which preserves mean=0 across samples.
      This is different from the Scikit-Learn scalers.
    - If axis=None, elements are X_trans
      using global geometric mean. Note that in such case
      X_trans data will not be zero-centered in any direction.
      
    Input
    -----
    pandas.DataFrame with rows (axis=0) as samples and 
    columns (axis=1) as feattures.

    Parameters
    ----------
    axis : int (0, 1 or None), default=1
        Specifies direction in which geometric mean is computed.
        If None, compute geometric mean globally (scalar value).

    TODO:
     - the method requires optimization (works too slow)
    """

    def __init__(self, axis=1):
        self.axis = axis
        # TODO fix it in a better way
        #https://stackoverflow.com/questions/21610198/
        #runtimewarning-divide-by-zero-encountered-in-log
        np.seterr(divide = 'ignore') 
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def fit(self, X):
        # Compute geometric mean for later transforming.
        X_masked = ma.masked_array(X, mask=[X == 0])
        self.gmean_ = gmean(X_masked, axis=self.axis)
        return self

    def transform(self, X):
        # Transform input data using geometric mean
        # computed with `self.fit()` method.
            
        if not self.axis:
            X_trans = np.log(X / self.gmean_, 
                             where=X != 0)
        else:
            X_trans = np.log(X.divide(self.gmean_, 
                                      axis=abs(self.axis-1)), 
                             where=X != 0)
        return X_trans

    def inverse_transform(self, X_trans, mask):
        # Compute inverse RCLR transform but only for
        # entries specified by boolean `mask` matrix
        # e.g. for values different from zero (by design).
        # Inverse RCLR transform (for non-zero geometric mean)
        # returns non zero values for zero inputs which is
        # desired only for inputs transformed into zero by 
        # RCLR transform i.e. their values were equal to 
        # geometric mean.
        # 
        if not self.axis:
            X = np.exp(X_trans, where=mask) * self.gmean_
        else:
            X = np.exp(X_trans, where=mask).multiply(
                self.gmean_, axis=abs(self.axis-1))
        return X
    

class CLRTransformer:
    """Transform features using Centered Log-Ratio 
    transformation with pseudocounts (CLR).
    
    - By defult, a table is X_trans column-wise (axis=1),
      which preserves mean=0 across samples.
      This is different from the Scikit-Learn scalers.
    - If axis=None, elements are X_trans
      using global geometric mean. Note that in such case
      X_trans data will not be zero-centered in any direction.
      
    Input
    -----
    numpy.array or pandas.DataFrame with
    rows (axis=0) as samples and columns (axis=1) as feattures.

    Parameters
    ----------
    axis : int (0, 1 or None), default=1
        Specifies direction in which geometric mean is computed.
        If None, compute geometric mean globally (scalar value).

    TODO:
     - the method requires optimization (works too slow)
    """

    def __init__(self, axis=1, pseudo_perc=0.01, 
                 is_pseudo_global=False):
        self.axis = axis
        self.pseudo_perc = pseudo_perc
        self.is_pseudo_global = is_pseudo_global
        # Pseudocounts can be added either globally or accross
        # the same axis which is used for geometric mean.
        if self.is_pseudo_global:
            self.pseudo_axis = None
        else:
            self.pseudo_axis = axis
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def fit(self, X):
        # Add pseudocounts to zero-valued entries
        # and then compute geometric mean for later transforming.
        X_nonzeros = ma.masked_array(X, mask=[X == 0])
        X_zeros = ma.masked_array(X, mask=[X != 0])
    
        self.min_ = np.min(X_nonzeros, axis=self.pseudo_axis) *\
        self.pseudo_perc

        if self.axis == 1:
            X_pseudo = (X_zeros.T + self.min_).T
        else:
            X_pseudo = X_zeros + self.min_
            
        self.gmean_ = gmean(X_pseudo.data, axis=self.axis)
        return self

    def transform(self, X):
        pass

    def inverse_transform(self, X_trans, mask):
        pass
    
    
def _test_RCLRTransformer():
    # TODO create unit test
    X = pd.DataFrame(np.array([[1, 0, 3, 6], 
                               [4, 5, 6, 7], 
                               [2, 5.5, 6, 8.2]]))

    Y = pd.DataFrame(np.array([[1, 0, 3, 6], 
                               [4, 5, 6,  7], 
                               [4, 0, 2,  0], 
                               [2, 5.5, 6,  8.2]]))
    
    expected_1 = np.array([[-0.96345725,  0.        ,  0.13515504,  0.82830222],
                           [-0.29705611, -0.07391256,  0.108409  ,  0.26255968],
                           [-0.88030004,  0.13130087,  0.21831225,  0.53068693]])

    expected_0 = np.array([[-0.69314718,  0.        , -0.46209812, -0.15550845],
                           [ 0.69314718, -0.04765509,  0.23104906, -0.00135778],
                           [ 0.        ,  0.04765509,  0.23104906,  0.15686623]])
    
    expected_None = np.array([[-1.44705114,  0.        , -0.34843885,  0.34470833],
                              [-0.06075678,  0.16238677,  0.34470833,  0.49885901],
                              [-0.75390396,  0.25769695,  0.34470833,  0.65708301]])

    # fit_transform
    transformer = RCLRTransformer()
    res_1 = transformer.fit_transform(X)
    assert np.allclose(res_1.values, expected_1)

    transformer = RCLRTransformer(axis=0)
    res_0 = transformer.fit_transform(X)
    assert np.allclose(res_0.values, expected_0)
   
    transformer = RCLRTransformer(axis=None)
    res_None = transformer.fit_transform(X)
    assert np.allclose(res_None.values, expected_None)

    # fit and transform
    transformer = RCLRTransformer()
    transformer.fit(X)
    res_1 = transformer.transform(X)
    assert np.allclose(res_1.values, expected_1)

    transformer = RCLRTransformer(axis=0)
    transformer.fit(X)
    res_0 = transformer.transform(X)
    assert np.allclose(res_0.values, expected_0)
    
    transformer = RCLRTransformer(axis=None)
    transformer.fit(X)
    res_None = transformer.transform(X)
    assert np.allclose(res_None.values, expected_None)
    
    # inverse_transform
    transformer = RCLRTransformer()
    res_1 = transformer.fit_transform(X)
    inv_res_1 = transformer.inverse_transform(res_1, X!=0)
    assert np.allclose(inv_res_1.values, X)

    transformer = RCLRTransformer(axis=0)
    res_0 = transformer.fit_transform(X)
    inv_res_0 = transformer.inverse_transform(res_0, X!=0)
    assert np.allclose(inv_res_0.values, X)
    
    transformer = RCLRTransformer(axis=None)
    res_None = transformer.fit_transform(X)
    inv_res_None = transformer.inverse_transform(res_None, X!=0)
    assert np.allclose(inv_res_None.values, X)
    print("RCLR tests passed.")
    

def _test_CLRTransformer():
    # TODO create unit test
    X = pd.DataFrame(np.array([[1, 0, 3, 6], 
                               [4, 5, 6, 0.7], 
                               [2, 5.5, 0, 8.2]]))

    # fit
    transformer = CLRTransformer()
    transformer.fit(X) 
    assert np.allclose(transformer.min_, [0.01, 0.007,0.02])
    
    transformer = CLRTransformer(axis=0)
    transformer.fit(X) 
    assert np.allclose(transformer.min_, [0.01, 0.05, 0.03, 0.007])
    
    transformer = CLRTransformer(axis=None)
    transformer.fit(X) 
    assert np.allclose(transformer.min_, 0.007)

    for axis in [0, 1, None]:
        transformer = CLRTransformer(axis=axis, pseudo_perc=0.05,
                                     is_pseudo_global=True)
        transformer.fit(X) 
        assert np.allclose(transformer.min_, 0.035)
    
    print("CLR tests passed.")
    
    
if __name__ == "__main__":
    
    _test_RCLRTransformer()
    _test_CLRTransformer()