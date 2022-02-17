import pandas as pd
import numpy as np

from scipy.stats import spearmanr
from scipy.spatial import procrustes
from scipy.spatial.distance import braycurtis
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import f1_score, mean_squared_error
from skbio.stats.ordination import pcoa


def continuous_to_multiclass(data):
    """Transform continuous data to multiclass"""
    data[data>0] = 1
    data[data<0] = 0
    return data


def calculate_f1score(true, pred, model=None, return_scalar=False):
    """Calculate f1_score across columns between true 
    and pred dataframes using continuous to multiclass
    transformation.

    Return
    ------
    - vector or scalar (vector mean)
    """
    true_m = continuous_to_multiclass(true.copy())
    pred_m = continuous_to_multiclass(pred.copy())
    res = true_m.combine(pred_m, f1_score).iloc[0]
    if return_scalar:
        return np.mean(res)
    else:
        return res.to_frame(model)
    

def calculate_spearman(true, pred, model=None, return_scalar=False):
    """Calculate a Spearman correlation coefficient
    across columns between true and pred dataframes.

    Return
    ------
    - vector or tuple (non-nan vector abs mean, number of nans)
    """
    coeffs = []
    for col in true.columns:
        rho, _ = spearmanr(true[col], pred[col])
        coeffs.append(rho)
    if return_scalar:
        # Returns tuple: (non-nan abs mean, number of nans)
        return (np.nanmean(np.abs(coeffs)), 
                np.count_nonzero(np.isnan(coeffs)))
    else:
        df = pd.DataFrame(coeffs, columns=[model], index=true.columns)
        return df
    
    
def calculate_nrmse(true, pred, model=None, return_scalar=False):
    """Calculate a normalized root mean squared error
    across columns between true and pred dataframes.

    Return
    ------
    - vector or tuple (non-nan vector mean, number of nans)
    
    TODO: optimize using pandas.DataFrame.combine()
    """
    coeffs = []
    for col in true.columns:
        divider = true[col].max() - true[col].min()
        nrmse = mean_squared_error(true[col], pred[col], squared=False)
        if divider != 0:
            coeffs.append(nrmse / divider)
        elif divider == 0 and nrmse != 0:
            coeffs.append(np.nan)
        elif divider == 0 and nrmse == 0:
            coeffs.append(0)
    if return_scalar:
        # Returns tuple: (non-nan mean, number of nans)
        return (np.nanmean(coeffs),
                p.count_nonzero(np.isnan(coeffs)))
    else:
        df = pd.DataFrame(coeffs, columns=[model], index=true.columns)
        return df
    
    
def inter_dissimilarity(true, pred, model=None, return_scalar=False):
    """Calculate Bray-Curtis inter dissimilarity
    across indices (samples) between true and pred dataframes.

    Return
    ------
    - vector or scalar (vector mean)
    """
    res = true.T.combine(pred.T, braycurtis).iloc[0]
    if return_scalar:
        return np.mean(res)
    else:
        return res.to_frame(model)
    

def intra_dissimilarity(true, pred, model):
    """Calculate Bray-Curtis intra dissimilarity
    across indices (samples) between true and pred dataframes
    then perform PCOA and procrustes analysis.
    
    Return
    ------
    - procrustes matrices (x, y) 
    - scalar (disparity)
    """
    true_m = squareform(pdist(true, metric='braycurtis'))
    pred_m = squareform(pdist(pred, metric='braycurtis'))
    true_ordination = pcoa(true_m).samples
    pred_ordination = pcoa(pred_m).samples
    x, y, disparity = procrustes(true_ordination, pred_ordination)
    return x, y, disparity


if __name__ == "__main__":
    pass