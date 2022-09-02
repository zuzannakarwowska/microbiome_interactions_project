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
                   in_steps=1, overlap=True, shuffle=True,
                   return_indices=False):
    """
    Split data into train/validation sets.
   
   `reframed` - output of the `series_to_supervised()` function.
   `cols` - number of features in the dataset.
   
    First, the data is reshaped into samples where each
    sample contains `in_steps` time points for
    trainining/validation. If `overlap=True`, each slice of data 
    is shifted by one row with respect to the subsequent one, which
    is important only for `in_steps>1`.
    
    Second, the data is shuffled, but if `shuffle=False`,
    then the test samples are taken from the end of the dataset.
    
    Finally, the data is split into train/test parts.
    
    Note: if you need to link the train/val data with the original dataframe,
    use `return_indices=True`.
    """
    # 1) Prepare indices list
    indices = reframed.index
    # skip last rows if needed
    if overlap:
        skipped = in_steps-1
    else:
        skipped = len(indices) % in_steps
    if skipped:
        indices = indices[:-skipped]
    # remove indices overlap
    if not overlap:
        indices = indices[::in_steps]

    # 2) Divide data into pieces and reshape
    if in_steps > 1:
        values = [reframed.loc[i:i+in_steps-1].values for i in indices]
    else:
        values = [reframed.loc[i].values for i in indices]
    # Only the first dimension will be shuffled (below)
    values = np.array(values).reshape(len(indices), in_steps, reframed.shape[1])

    # 3) Shuffle data
    # Note 1: original input dataframe will be shuffled as well !!!!
    # Note 2: it concerns only the first dimension (i.e. groups of 
    #`in_steps x cols` elements)
    order = np.array(range(len(indices)))
    if shuffle:
        np.random.shuffle(order)
        values = values[order]
        indices = indices[order]

    # 4) Split data
    all_features = cols * in_steps
    train_last_id = int(values.shape[0] * train_test_split)
    train_X, train_y = values[:train_last_id, :, :all_features], \
    values[:train_last_id, :, all_features:]
    test_X, test_y = values[train_last_id:, :, :all_features], \
    values[train_last_id:, :, all_features:]
    # Use only first time step for y
    train_y = train_y[:, 0, :] 
    test_y = test_y[:, 0, :]
    # Use only the first set of columns (t - in_steps) for X
    # (using everything would introduce redundancy)
    train_X = train_X[:, :, :cols] 
    test_X = test_X[:, :, :cols]

    train_indices_y = indices[:train_last_id]
    test_indices_y = indices[train_last_id:]
    assert len(values) == (train_X.shape[0] + test_X.shape[0]) 
    assert (train_y[:,:2] == reframed.loc[train_indices_y, 
                                          ['var1(t)', 'var2(t)']].values).all()
    assert (test_y[:,:2] == reframed.loc[test_indices_y, 
                                         ['var1(t)', 'var2(t)']].values).all()
    if return_indices:
        return train_X, train_y, test_X, test_y, train_indices_y, test_indices_y
    else:
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
    
    If order='C', put time steps one after another:
    e.g. var1(t-2), var2(t-2), var1(t-1), var2(t-1)
    
    If order='F', put features one after another:
    e.g. var1(t-2), var1(t-1), var2(t-2), var2(t-1)
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