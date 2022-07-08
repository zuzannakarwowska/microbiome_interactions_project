import pandas as pd
import numpy as np
import warnings

from scipy.stats import spearmanr
from scipy.spatial import procrustes
from scipy.spatial.distance import braycurtis
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import f1_score, mean_squared_error
from skbio.stats.ordination import pcoa


"""Codebase for this module was delivered by 
https://github.com/zuzannakarwowska"""


def check_inputs(true, pred):
    assert type(true) is pd.DataFrame, \
    'Frist input is not of type pandas.DataFrame'
    assert type(pred) is pd.DataFrame, \
    'Second input is not of type pandas.DataFrame'
    assert true.shape == pred.shape, \
    f'Inputs have different shape: {true.shape} ! {pred.shape}'
    
    
def continuous_to_multiclass(data):
    """Transform continuous data to multiclass"""
    data[data>0] = 1
    data[data<0] = 0
    return data


def calculate_f1score(true, pred, model=None, axis=0, 
                      return_tuple=False):
    """Calculate f1_score across axis between true 
    and pred dataframes using continuous to multiclass
    transformation.

    Return
    ------
    - vector or tuple (vector's mean, vector's standard deviation)
    """
    check_inputs(true, pred)
    true_m = continuous_to_multiclass(true.copy())
    pred_m = continuous_to_multiclass(pred.copy())
    if axis == 0:
        true_m = true_m.T
        pred_m = pred_m.T
    res = true_m.combine(pred_m, f1_score).iloc[0]
    if return_tuple:
        return (np.mean(res), np.std(res))
    else:
        return res.to_frame(model)
    

def calculate_spearman(true, pred, model=None, return_tuple=False):
    """Calculate a Spearman correlation coefficient
    across columns between true and pred dataframes.

    Return
    ------
    - vector or tuple (non-nan vector's abs mean, number of nans)
    """
    check_inputs(true, pred)
    coeffs = []
    with warnings.catch_warnings():
        # Supress SpearmanRConstantInputWarning
        # (number of NaNs is returned anyway)
        warnings.simplefilter("ignore")
        for col in true.columns:
            rho, _ = spearmanr(true[col], pred[col])
            coeffs.append(rho)
    if return_tuple:
        # Returns tuple: (non-nan abs mean, non-nan abs std, 
        #                number of nans)
        return (np.nanmean(np.abs(coeffs)), 
                np.nanstd(np.abs(coeffs)), 
                np.count_nonzero(np.isnan(coeffs)))
    else:
        df = pd.DataFrame(coeffs, columns=[model], index=true.columns)
        return df
    
    
def calculate_nrmse(true, pred, model=None, return_tuple=False,
                   div_rtol=1e-2, nrmse_rtol=1):
    """Calculate a normalized root mean squared error
    across columns between true and pred dataframes.

    Return
    ------
    - vector or tuple (non-nan vector's mean, number of nans)
    
    Notes
    -----
    `div_rtol = 1e-2` is suitabe for defualt pseudocount threshold
    `nrmse_rtol = 1` is suitabe for comparing counts
    
    TODO: optimize using pandas.DataFrame.combine()
    """
    check_inputs(true, pred)
    coeffs = []
    for col in true.columns:
        divider = true[col].max() - true[col].min()
        nrmse = mean_squared_error(true[col], pred[col], squared=False)
        if abs(divider) >= div_rtol:
            coeffs.append(nrmse / divider)
        elif abs(divider) < div_rtol and nrmse >= nrmse_rtol:
            coeffs.append(np.nan)
            print(nrmse)
        elif abs(divider) < div_rtol and nrmse < nrmse_rtol:
            coeffs.append(0)
    if return_tuple:
        # Returns tuple: (non-nan mean, non-nan std, number of nans)
        return (np.nanmean(coeffs), np.nanstd(coeffs),
                np.count_nonzero(np.isnan(coeffs)))
    else:
        df = pd.DataFrame(coeffs, columns=[model], index=true.columns)
        return df
    
    
def inter_dissimilarity(true, pred, model=None, return_tuple=False):
    """Calculate Bray-Curtis inter dissimilarity
    across indices (samples) between true and pred dataframes.

    Return
    ------
    - vector or tuple (vector's mean, vector's standard deviation)
    """
    check_inputs(true, pred)
    res = true.T.combine(pred.T, braycurtis).iloc[0]
    if return_tuple:
        return (np.mean(res), np.std(res))
    else:
        return res.to_frame(model)
    

def intra_dissimilarity(true, pred):
    """Calculate Bray-Curtis intra dissimilarity
    across indices (samples) between true and pred dataframes
    then perform PCOA and procrustes analysis.
    
    Return
    ------
    - two PCoA ordination Pandas dataframes: 
      `true_ordination`, `pred_ordination` 
    - two procrustes NumPy arrays: `x`, `y` 
    - scalar: `disparity`
    """
    check_inputs(true, pred)
    true_m = squareform(pdist(true, metric='braycurtis'))
    pred_m = squareform(pdist(pred, metric='braycurtis'))
    true_ordination = pcoa(true_m).samples
    pred_ordination = pcoa(pred_m).samples
    x, y, disparity = procrustes(true_ordination, pred_ordination)
    return true_ordination, pred_ordination, x, y, disparity


if __name__ == "__main__":
    pass