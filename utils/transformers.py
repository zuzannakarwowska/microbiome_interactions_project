import numpy as np
import pandas as pd
import numpy.ma as ma

from scipy.stats import gmean
from sklearn.preprocessing import MinMaxScaler
    
    
#=========
# Helpers
#=========

def _transform_type(X):
    if type(X) is not pd.DataFrame:
            return pd.DataFrame(X)
    else:
        return X

    
def inverse_through_timesteps_wrapper(scaler, dataset, predictions, sparams):
    """
    Use this wrapper if your scaler inverse transform is performed
    on timesteps e.g. clr-1.
    """
    # Merge all predictions into one dataframe
    y = pd.concat([pd.DataFrame(**d) for d in predictions])
    # Adjust index (timesteps) to match the dataset index
    y = y.sort_index().reindex(dataset.index)
    # Inverse transform
    inv_y = scaler.inverse_transform(y, **sparams)
    # Return inverted predictions
    inv_y_sets = []
    for d in predictions:
        inv_y_sets.append(inv_y.loc[d['index']])
    return inv_y_sets        


#==============
# Transformers
#==============

class RCLRTransformer:
    """Transform features using Robust Centered Log-Ratio 
    transformation (RCLR).
    
    - By defult, a table is transformed column-wise (axis=1),
      which preserves mean=0 across samples.
      This is different from the Scikit-Learn scalers.
    - If axis=None, elements are transformed
      using global geometric mean. Note that in such case
      X_trans data will not be zero-centered in any direction.
      
    Input
    -----
    pandas.DataFrame with rows (axis=0) as samples and 
    columns (axis=1) as features.

    Parameters
    ----------
    axis : int (0, 1 or None), default=1
        Specifies direction in which geometric mean is computed.
        If None, compute geometric mean globally (scalar value).
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
        X = _transform_type(X)
        if self.axis is None:
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
        X_trans = _transform_type(X_trans)
        if self.axis is None:
            X = np.exp(X_trans, where=mask) * self.gmean_
        else:
            X = np.exp(X_trans, where=mask).multiply(
                self.gmean_, axis=abs(self.axis-1))
        return X
    

class CLRTransformer:
    """Transform features using Centered Log-Ratio 
    transformation with pseudocounts (CLR) (see [1]).
    
    - By defult, a table is transformed column-wise (axis=1),
      which preserves mean=0 across samples.
      This is different from the Scikit-Learn scalers.
    - If axis=None, elements are transformed
      using global geometric mean. Note that in such case
      X_trans data will not be zero-centered in any direction.
      
    Input
    -----
    pandas.DataFrame with rows (axis=0) as samples and 
    columns (axis=1) as features.
    
    Parameters
    ----------
    axis : int (0, 1 or None), default=1
        Specifies direction in which geometric mean is computed.
        If None, compute geometric mean globally (scalar value).
        
    pseudo_mode : str, 'perc' or 'value', default='perc'
        Either to use the `pseudo_value` as absolute pseudocount value 
        (choose 'value') or as percentage of a minimum value in the input
        table / axis (choose 'perc').
        
    pseudo_value : float, default=0.01
        Absolute value or percentage of a minimum value in the input
        table / axis which is then used as a pseudocunt (see also a 
        `pseudo_mode` parameter above).
        
    is_pseudo_global : bool, default=False
        If pseudocount is a global value (scaled minimum of the 
        whole input table). If False, then compute minimum
        using the `axis` parameter.
        
    add_pseudo_to_zeros_only : bool, default=True
        If pseudocounts should be added everywhere (False) or 
        only to zero entries (True).
        
    References
    ----------
    [1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6755255/
    """

    def __init__(self, axis=1, pseudo_mode='perc', pseudo_value=0.01, 
                 is_pseudo_global=False, add_pseudo_to_zeros_only=True):
        self.axis = axis
        self.pseudo_mode = pseudo_mode
        self.pseudo_value = pseudo_value
        self.is_pseudo_global = is_pseudo_global
        self.add_pseudo_to_zeros_only = add_pseudo_to_zeros_only
        # Pseudocounts can be added either globally or accross
        # the same axis which is used for geometric mean.
        if self.is_pseudo_global:
            self.pseudo_axis = None
        else:
            self.pseudo_axis = axis
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _add_pseudocounts(self, X):
        if self.add_pseudo_to_zeros_only:
            # Add pseudocounts only to zero-valued entries in X
            masks = [[X != 0], [X == 0]]
        else:
            masks = [False, False]
        X_zeros = ma.masked_array(X, mask=masks[0])
        X_nonzeros = ma.masked_array(X, mask=masks[1])
        # Compute pseudocount
        if self.pseudo_mode == 'perc':
            self.pseudo_count_ = np.min(X_nonzeros, 
                               axis=self.pseudo_axis) * self.pseudo_value
        elif self.pseudo_mode == 'value':
            self.pseudo_count_ = self.pseudo_value   
        else:
            raise ValueError(f'self.pseudo_mode cannot be {self.pseudo_mode}')
        # Add pseudocount
        if self.pseudo_axis == 1:
            X_pseudo = (X_zeros.T + self.pseudo_count_).T
        else:
            X_pseudo = X_zeros + self.pseudo_count_
        return X_pseudo   
    
    def fit(self, X):
        # Compute and add pseudocounts to zero-valued entries
        # and then compute geometric mean for later transforming.
        X_pseudo = self._add_pseudocounts(X)
        self.gmean_ = gmean(X_pseudo.data, axis=self.axis)
        return self 

    def transform(self, X):
        # Transform input data using geometric mean
        # computed with `self.fit()` method.
        X = _transform_type(X)
        X_trans = self._add_pseudocounts(X)
        X_trans = pd.DataFrame(data=X_trans.data, 
                       index=X.index, columns=X.columns)
        if self.axis is None:
            X_trans = np.log(X_trans / self.gmean_)
        else:
            X_trans = np.log(X_trans.divide(self.gmean_, 
                                      axis=abs(self.axis-1)))
        return X_trans

    def inverse_transform(self, X_trans, mask=None, 
                          remove_pseudocounts=True):
        # Compute inverse CLR transform.
        # `mask` is a boolean matrix where False
        # specifies pseudocunt position.
        # If `remove_pseudocounts=False`, then
        # the `mask` parameter is not required.
        X_trans = _transform_type(X_trans)
        if self.axis is None:
            X = np.exp(X_trans) * self.gmean_
            if remove_pseudocounts:
                if self.add_pseudo_to_zeros_only:
                    X -= ~mask * self.pseudo_count_
                else:
                    X -= self.pseudo_count_
        else:
            X = np.exp(X_trans).multiply(
                self.gmean_, axis=abs(self.axis-1))
            if remove_pseudocounts:
                if self.add_pseudo_to_zeros_only:
                    X -= pd.DataFrame(~mask).multiply(self.pseudo_count_, 
                                                      axis=abs(self.axis-1))
                else:
                    X -= pd.DataFrame(data=True, index=X.index, 
                                      columns=X.columns).multiply(
                        self.pseudo_count_, axis=abs(self.axis-1))               
        return X
    

class Log1pMinMaxScaler:
    """Transform features using log(1+X) and then scale 
    each feature to a given range.

    Input
    -----
    numpy.array or pandas.DataFrame with rows (axis=0) 
    as samples and columns (axis=1) as features.

    Parameters
    ----------
    The same as for sklearn.preprocessing.MinMaxScaler.
    """
    
    def __init__(self, *args, **kwargs):
        # Create a new min-max scaler
        self.minmax_scaler = MinMaxScaler(*args, **kwargs)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def fit(self, X):
        # Fit the min-max scaler on log(1+X)
        self.minmax_scaler.fit(np.log1p(X))
        return self

    def transform(self, X):
        X_trans = self.minmax_scaler.transform(np.log1p(X))
        return X_trans

    def inverse_transform(self, X_trans):
        X = self.minmax_scaler.inverse_transform(X_trans)
        X = np.exp(X) - 1
        return X

    
class IdentityScaler:
    """Does nothing with the input data.

    Input
    -----
    numpy.array or pandas.DataFrame with rows (axis=0) 
    as samples and columns (axis=1) as features.
    """
    
    def __init__(self):
        pass
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X
    
    
#=======
# Tests
#=======
    
def _test_RCLRTransformer():
    X = pd.DataFrame(np.array([[1, 0, 3, 6], 
                               [4, 5, 6, 7], 
                               [2, 5.5, 6, 8.2]]))
    
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
    X = pd.DataFrame(np.array([[1, 0, 3, 6], 
                               [4, 5, 6, 0.7], 
                               [2, 5.5, 0, 8.2]]))

    # Test `pseudo_axis` and `is_pseudo_global` parameters
    # for `pseudo_mode='perc'` and `add_pseudo_to_zeros_only=True`
    
    expected_1_False = np.array([[ 0.42869961, -4.17647058,  1.5273119 ,  2.22045908],
                                 [ 0.27859016,  0.50173371,  0.68405527, -1.46437914],
                                 [ 0.54564558,  1.55724649, -4.05952461,  1.95663255]])

    expected_0_False = np.array([[-0.69314718, -3.10188352,  1.30400767,  0.61201991],
                                 [ 0.69314718,  1.50328667,  1.99715485, -1.5364145 ],
                                 [ 0.        ,  1.59859685, -3.30116252,  0.92439459]])
    
    expected_1_True = np.array([[ 0.51786834, -4.44397679,  1.61648063,  2.30962781],
                                [ 0.27859016,  0.50173371,  0.68405527, -1.46437914],
                                [ 0.80810111,  1.81970202, -4.8468912 ,  2.21908808]])

    expected_0_True = np.array([[-0.69314718, -4.41262542,  1.78910341,  0.61201991],
                                [ 0.69314718,  2.15865762,  2.48225059, -1.5364145 ],
                                [ 0.        ,  2.2539678 , -4.27135401,  0.92439459]])
    
    expected_None = np.array([[-0.15829398, -5.12013911,  0.94031831,  1.63346549],
                              [ 1.22800038,  1.45114394,  1.63346549, -0.51496892],
                              [ 0.5348532 ,  1.54645412, -5.12013911,  1.94584018]])

    # fit
    transformer = CLRTransformer()
    transformer.fit(X) 
    assert np.allclose(transformer.pseudo_count_, [0.01, 0.007,0.02])
    
    transformer = CLRTransformer(axis=0)
    transformer.fit(X) 
    assert np.allclose(transformer.pseudo_count_, [0.01, 0.05, 0.03, 0.007])
    
    transformer = CLRTransformer(axis=None)
    transformer.fit(X) 
    assert np.allclose(transformer.pseudo_count_, 0.007)

    for axis in [0, 1, None]:
        transformer = CLRTransformer(axis=axis, pseudo_value=0.05,
                                     is_pseudo_global=True)
        transformer.fit(X) 
        assert np.allclose(transformer.pseudo_count_, 0.035)
    
    # fit transform
    transformer = CLRTransformer()
    transformer.fit(X) 
    res_1_False = transformer.transform(X) 
    assert np.allclose(res_1_False, expected_1_False)

    transformer = CLRTransformer(axis=0)
    transformer.fit(X) 
    res_1_False = transformer.transform(X) 
    assert np.allclose(res_1_False, expected_0_False)

    transformer = CLRTransformer(axis=None)
    transformer.fit(X) 
    res_None_False = transformer.transform(X) 
    assert np.allclose(res_None_False, expected_None)

    transformer = CLRTransformer(is_pseudo_global=True)
    transformer.fit(X) 
    res_1_True = transformer.transform(X) 
    assert np.allclose(res_1_True, expected_1_True)
    
    transformer = CLRTransformer(axis=0, is_pseudo_global=True)
    transformer.fit(X) 
    res_0_True = transformer.transform(X) 
    assert np.allclose(res_0_True, expected_0_True)
    
    transformer = CLRTransformer(axis=None, is_pseudo_global=True)
    transformer.fit(X) 
    res_None_True = transformer.transform(X) 
    assert np.allclose(res_None_True, expected_None)
    
    # inverse transform
    transformer = CLRTransformer()
    res_1_False = transformer.fit_transform(X)
    inv_res_1_False = transformer.inverse_transform(res_1_False, X!=0)
    assert np.allclose(inv_res_1_False.values, X)

    transformer = CLRTransformer(axis=0)
    res_0_False = transformer.fit_transform(X)
    inv_res_0_False = transformer.inverse_transform(res_0_False, X!=0)
    assert np.allclose(inv_res_0_False.values, X)
    
    transformer = CLRTransformer(axis=None)
    res_None_False = transformer.fit_transform(X)
    inv_res_None_False = transformer.inverse_transform(res_None_False, X!=0)
    assert np.allclose(inv_res_None_False.values, X)
    
    transformer = CLRTransformer(is_pseudo_global=True)
    res_1_True = transformer.fit_transform(X)
    inv_res_1_True = transformer.inverse_transform(res_1_True, X!=0)
    assert np.allclose(inv_res_1_True.values, X)

    transformer = CLRTransformer(axis=0, is_pseudo_global=True)
    res_0_True = transformer.fit_transform(X)
    inv_res_0_True = transformer.inverse_transform(res_0_True, X!=0)
    assert np.allclose(inv_res_0_True.values, X)
    
    transformer = CLRTransformer(axis=None, is_pseudo_global=True)
    res_None_True = transformer.fit_transform(X)
    inv_res_None_True = transformer.inverse_transform(res_None_True, X!=0)
    assert np.allclose(inv_res_None_True.values, X)

    # test `add_pseudo_to_zeros_only` parameter
    # for `pseudo_mode='value'` and `is_pseudo_global=True`

    expected = np.array([[-0.30845271, -1.34454465,  0.52023996,  1.1327574 ],
                         [ 0.18224379,  0.38091449,  0.54658161, -1.10973989],
                         [-0.14074872,  0.72321619, -1.67467908,  1.09221162]])
 
    transformer = CLRTransformer(axis=1, is_pseudo_global=True, 
                                 pseudo_mode='value', pseudo_value=0.55, 
                                 add_pseudo_to_zeros_only=False)
    transformer.fit(X) 
    res = transformer.transform(X) 
    assert np.allclose(res, expected)
    res = transformer.fit_transform(X)
    inv_res = transformer.inverse_transform(res, remove_pseudocounts=False)
    assert np.allclose(inv_res, X + 0.55)
    inv_res = transformer.inverse_transform(res, remove_pseudocounts=True)
    assert np.allclose(inv_res, X)

    expected = np.array([[-0.57313369, -1.17097069,  0.5254786 ,  1.21862578],
                         [ 0.27859016,  0.50173371,  0.68405527, -1.46437914],
                         [-0.28290093,  0.72869999, -1.57388511,  1.12808605]])

    expected_inv_False = np.array([[1, 0.55, 3, 6], 
                                   [4, 5, 6, 0.7], 
                                   [2, 5.5, 0.55, 8.2]])
    
    transformer = CLRTransformer(axis=1, is_pseudo_global=True, 
                                 pseudo_mode='value',  pseudo_value=0.55, 
                                 add_pseudo_to_zeros_only=True)
    transformer.fit(X) 
    res = transformer.transform(X) 
    assert np.allclose(res, expected)
    res = transformer.fit_transform(X)
    inv_res = transformer.inverse_transform(res, remove_pseudocounts=False)
    assert np.allclose(inv_res, expected_inv_False)
    inv_res = transformer.inverse_transform(res, X!=0, remove_pseudocounts=True)
    assert np.allclose(inv_res, X)

    print("CLR tests passed.")
    

def _test_Log1pMinMaxScaler():
    X = pd.DataFrame(np.array([[1, 0, 3, 6], 
                               [4, 5, 6, 0.7], 
                               [2, 5.5, 0, 8.2]]))
    
    expected = np.array([[0., 0., 0.71241437, 0.83815152],
                         [1., 0.95723762, 1., 0.],
                         [0.44250705, 1., 0., 1.]])
    
    # fit transform
    scaler = Log1pMinMaxScaler()
    scaler.fit(X)
    res = scaler.transform(X)
    assert np.allclose(res, expected)
    
    # inverse transform
    scaler = Log1pMinMaxScaler(feature_range=[-1, 2])
    res = scaler.fit_transform(X)
    inv_res = scaler.inverse_transform(res)
    assert np.allclose(inv_res, X)

    print("Log1pMinMax tests passed.")
    

def _test_IdentityScaler():
    X = pd.DataFrame(np.array([[1, 0, 3, 6], 
                               [4, 5, 6, 0.7], 
                               [2, 5.5, 0, 8.2]]))
    
    # fit transform
    scaler = IdentityScaler()
    scaler.fit(X)
    res = scaler.transform(X)
    assert np.allclose(res, X)
    
    # inverse transform
    scaler = IdentityScaler()
    res = scaler.fit_transform(X)
    assert np.allclose(res, X)
    inv_res = scaler.inverse_transform(res)
    assert np.allclose(inv_res, X)

    print("Identity tests passed.")
    
    
if __name__ == "__main__":
    
    _test_RCLRTransformer()
    _test_CLRTransformer()
    _test_Log1pMinMaxScaler()
    _test_IdentityScaler()