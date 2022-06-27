import pandas as pd
import random 
import numpy as np
import numpy.ma as ma
import statsmodels.api as sm

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
