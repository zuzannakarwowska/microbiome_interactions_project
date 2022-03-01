import pandas as pd
import numpy as np


"""Some pieces borrowed from:
https://machinelearningmastery.com/"""


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Transform multivariate time series (with rows as time steps
    and columns as features) into supervised dataset.

    n_in: number of time steps used for prediction
    n_out: number of time steps to be predicted.

    If n_in > 1, the result dataframe contains some redundant
    information (some columns, e.g. var(t-2) and var(t-1), are the
    the same but shifted by one or more rows).
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def split_reframed(reframed, cols, train_test_split=0.8, 
                   no_of_timesteps=1, shuffle=True):
    """
    Split data into train/validation sets.
   
   `reframed` - output of the `series_to_supervised()` function.
   `cols` - number of features in the dataset.
   
    First, the data is reshaped into samples where each
    sample contains `no_of_timesteps` time points for
    trainining/validation.
    
    Second, the data is shuffled, but if `shuffle=False`,
    then the test samples are taken from the end of the dataset.
    
    Finally, the data is split into train/test parts.
    
    # TODO: consider an overlap between samples !!!!
    """
    # We need to skip last rows in order to perform reshaping
    values_to_throw = len(reframed) % no_of_timesteps
    values = reframed.values
    if values_to_throw:
        values = values[:-values_to_throw]
    # 1) Reshape data
    # Only the first dimension will be shuffled (below)
    values_reshaped = values.reshape(-1, no_of_timesteps, reframed.shape[1])
    # 2) Shuffle data
    # Note 1: original input dataframe will be shuffled as well !!!!
    # Note 2: it concerns only the first dimension (i.e. groups of 
    #`no_of_timesteps x cols` elements)
    if shuffle:
        np.random.shuffle(values_reshaped)
    # 3) Split data
    all_features = cols * no_of_timesteps
    train_last_id = int(values_reshaped.shape[0] * train_test_split)
    train_X, train_y = values_reshaped[:train_last_id, :, :all_features], \
    values_reshaped[:train_last_id, :, all_features:]
    test_X, test_y = values_reshaped[train_last_id:, :, :all_features], \
    values_reshaped[train_last_id:, :, all_features:]
    # Use only first time step for y
    train_y = train_y[:, 0, :] 
    test_y = test_y[:, 0, :]
    # Use only the first set of columns (t - no_of_timesteps) for X
    # (using everything would introduce redundancy)
    train_X = train_X[:, :, :cols] 
    test_X = test_X[:, :, :cols]
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # Check if all dimensions agree
    assert len(reframed) == (train_X.shape[0] + test_X.shape[0]) * \
    no_of_timesteps + values_to_throw    
    return train_X, train_y, test_X, test_y


def prepare_sequential_data(train_X, test_X=None, in_features=1):
    """
    Prepare list of training features.
    Use it if your model requires multiple inputs.
    """
    train_X_split = [train_X[:, :, i] for i in range(in_features)]
    if test_X is not None:
        test_X_split = [test_X[:, :, i] for i in range(in_features)]
    else:
        # Used for inference
        test_X_split = None
    return train_X_split, test_X_split


def prepare_supervised_data(train_X, test_X=None, order='C'):
    """
    Reshape sequential data into supervised features.
    Use it if your model requires one vector-like input.
    
    If order='C', put features one after another:
    e.g. var1(t-2), var1(t-1), var2(t-2), var1(t-1)
    
    If order='F', put time steps one after another:
    e.g. var1(t-2), var2(t-2), var1(t-1), var2(t-1)
    """
    if len(train_X.shape) == 3:
        train_X_reshaped = train_X.reshape((train_X.shape[0], -1), 
                                           order=order)
        if test_X is not None:
            test_X_reshaped = test_X.reshape((test_X.shape[0], -1), 
                                               order=order)
        else:
            # Used for inference
            test_X_reshaped = None
        return train_X_reshaped, test_X_reshaped
    else:
        print("Nothing done. Data are already reshaped.")
        return train_X, test_X
    
    

if __name__ == '__main__':
    pass