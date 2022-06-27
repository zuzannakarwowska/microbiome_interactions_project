import pandas as pd
import random 
import numpy as np
import numpy.ma as ma
import statsmodels.api as sm
from scipy.stats import gmean

def _transform_type(X):
        if type(X) is not pd.DataFrame:
                return pd.DataFrame(X)
        else:
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
        
    pseudo_perc : float, default=0.01
        Percentage of a minimum value in the input table which is
        then used as a pseudocunt.
        
    is_pseudo_global : bool, default=False
        If pseudocount is a global value (scaled minimum of the 
        whole input table). If False, then compute minimum
        using the `axis` parameter.
        
    References
    ----------
    [1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6755255/
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

    def _add_pseudocounts(self, X):
        # Add pseudocounts to zero-valued entries in X.
        X_zeros = ma.masked_array(X, mask=[X != 0])
        if self.axis == 1:
            X_pseudo = (X_zeros.T + self.min_).T
        else:
            X_pseudo = X_zeros + self.min_
        return X_pseudo   
    
    def fit(self, X):
        # Compute and add pseudocounts to zero-valued entries
        # and then compute geometric mean for later transforming.
        X_nonzeros = ma.masked_array(X, mask=[X == 0])
        self.min_ = np.min(X_nonzeros, 
                           axis=self.pseudo_axis) * self.pseudo_perc
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
                X -= ~mask * self.min_
        else:
            X = np.exp(X_trans).multiply(
                self.gmean_, axis=abs(self.axis-1))
            if remove_pseudocounts:
                X -= pd.DataFrame(~mask).multiply(self.min_, 
                                                  axis=abs(self.axis-1))
        return 
    

class MicrobiomeDataPreprocessing:
    
    '''
    prepare microbiome data for regression models 
    '''
    
    def filter_rare_features(self, df, treshold_perc=.9):
        
         
        ''' 
        filter bacteria that are absent in treshold_perc timepoints

        Params
        ---------
        df            - dataframe with samples in rows and features 
                         in columns
        treshold_perc - % timepoints in which bacteria needs to be 
                         absent to be filtered

        Returns
        ---------
        df_filtered   - filtered dataframe 
         
        '''
    
        treshold = round(df.shape[0] * treshold_perc)
        rare_bacteria_df = pd.DataFrame((df == 0).astype(int).sum(axis = 0))
        rare_bacteria_col = rare_bacteria_df[rare_bacteria_df[0] > treshold].index
        df_filtered = df.drop(rare_bacteria_col, axis = 1)

        return df_filtered
    
    def filter_random_features(self, df, lags=30):

        '''

        WHITE NOISE

        A time series is white noise if the variables are independent
        and identically distributed with a mean of zero. This means
        that all variables have the same variance (sigma^2) and each
        value has a zero correlation with all other values in the 
        series.

        We want to get rid of those series as they canot be modeled.

        For this we use The Ljung-Box test for white noise detection, 
        where
        H0: serie is white noise (p>0.05)
        H1: serie is not white noise (p<0.05)


        RANDOM WALK

        A random walk is another time series model where the current
        observation is equal to the previous observation with a
        random step up or down. To test wether series in RW we will 
        difference series and test for white noise behaviour with
        Ljung-Box test.

        Params
        -------

        df   - dataframe
        lags - lags used for autocorrelation test in Ljung-Box test

        Returns
        -------

        filtered_df - dataframe with removed white noise and random 
        walk features
        WHITE_NOISE_ASV - featrues exhibiting white noise behaviour
        RANDOM_WALK_ASV - featrues exhibiting random walk behaviour

        '''
    
        WHITE_NOISE_ASV = []
        RANDOM_WALK_ASV = []

        for i in range(0, df.shape[1]):

            x = df.iloc[:, i]

            #detect white noise
            res = sm.stats.acorr_ljungbox(x, lags=30, return_df=True)
            pvalues = res[res['lb_pvalue']> 0.05]

            if pvalues.shape[0] != 0:

                WHITE_NOISE_ASV.append(df.iloc[:, i].name)

            elif pvalues.shape[0] == 0: 

                diff_x = x.diff().dropna()
                diff_res = sm.stats.acorr_ljungbox(diff_x, lags=30, return_df=True)
                diff_pvalues = diff_res[diff_res['lb_pvalue']> 0.05]

                if diff_pvalues.shape[0] != 0:
                    RANDOM_WALK_ASV.append(df.iloc[:, i].name)
            else:
                pass

        filtered_df = df.drop(columns = WHITE_NOISE_ASV)
        filtered_df = filtered_df.drop(columns = RANDOM_WALK_ASV)


        return filtered_df, WHITE_NOISE_ASV, RANDOM_WALK_ASV

    
    def make_supervised(self, df, maxlag):
        
        '''  
        transform dataframe into a supervised problem 

        Params
        ---------
        df     - dataframe with samples in rows and features 
                 in columns
        maxlag - how many previous timesteps to use

        Returns
        ---------
        lagged_df   - supervised dataframe 

        '''
            
    
        df_list = []
        for i in range(0, maxlag+1):
            shifted_df = df.shift(i)
            shifted_df.columns = [col + '_lag{}'.format(i) for col in df.columns]
            df_list.append(shifted_df)

        lagged_df = pd.concat(df_list, axis=1)
        lagged_df = lagged_df.iloc[maxlag:]

        return lagged_df
    
    
class MicrobiomeTraintestSplit:
    
    
    ''' 
    Split microbiome time series data into 
    train and test
    
    Params
    ---------
    prc_split - n% of timesteps in series to be
    used as test set
    '''
        
    def __init__(self, prc_split=.1):
        
        self.prc_split = prc_split

    def last_block_split(self, df):
        
        '''  
        split supervised dataframe into train and 
        test where test is the last n% of timesteps
        in series
        

        Params
        ---------
        df     - dataframe with samples in rows and 
        features in columns
        prc_split - n% of timesteps in series to be
        used as test set

        Returns
        ---------
        train   -  dataframe with train set
        test    -  dataframe with test set

        '''
    
        split = int(df.shape[0] * self.prc_split) 

        train_set = df.iloc[:-split]
        test_set = df.iloc[-split:]

        return train_set, test_set
    
    ###############################################################################
    
    def blocked_split(self, df, chunk_size=1):
        
        '''  
        split supervised dataframe into train and 
        test where test are randomly selected chunks
        of size chunk_size. 
        

        Params
        ---------
        df         - dataframe with samples in rows and 
                     features in columns
        chunk_size - chunks into which df is split. Number of consecutive
                     timsteps
        prc_split  - what % of df we want to use
                     for test 

        Returns
        ---------
        train      -  dataframe with train set
        test       -  dataframe with test set
        
        
        Ex. with default settings dataframe will be split 
        into chunks of size 1 and 10% of dataframe length 
        will be used as test set. Mind if chunk_size is equal
        to 2, test_set size will be prc_split*chunk_size
        
        '''
    
        random.seed(0)
        #size of chunk
        chunk_size = chunk_size
        #define how many points to use in test
        test_size = int(np.round(len(df)*self.prc_split))

        #define list
        idx_list = df.index.tolist()

        #split index into chunks of size chunk_size
        chunks_list = [idx_list[i:i + chunk_size] for i in range(0, len(idx_list), chunk_size)]

        #randomly choose test chunks 
        test_idx = random.sample(chunks_list[5:], test_size)
        # define train chunks 
        train_idx = [i for i in chunks_list if i not in test_idx]

        #fltten
        test_idx = [x for xs in test_idx for x in xs]
        train_idx = [x for xs in train_idx for x in xs]

        #filter df 
        test_set = df[df.index.isin(test_idx)]
        train_set = df[df.index.isin(train_idx)]
    
    
        return train_set, test_set