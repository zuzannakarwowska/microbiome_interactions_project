import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.layers.merge import concatenate
from tensorflow.keras.regularizers import L1L2


class naive_predictor:
    """Return input as prediction.
    
    Input
    -----
    numpy.array or pandas.DataFrame with rows (axis=0) 
    as samples and columns (axis=1) as features.

    Parameters
    ----------
    type_: either 'sup' or 'seq'
        Type of MLP predictor (supervised or
        sequential) to be mimicked.
        
    n_in: int, default=1
        Number of input timesteps.

    n_out: int, default=1
        Number of output timesteps.
        
    n_cols: int
        Number of last columns to be returned. 
        
    TODO: doesn't work yet for n_out > 1
    """
    
    def __init__(self, type_, cols, n_in=1, n_out=1):
        self.type_ = type_
        self.cols = cols
        self.n_in = n_in
        self.n_out = n_out
        
    def count_params(self):
        return 0
 
    def predict(self, X):
        if self.n_out < 0 or self.n_in < 0:
            raise ValueError
        elif self.n_out > 1:
            raise NotImplementedError
        else:
            # Return last `cols` entries
            if self.type_ == 'sup':
                return X[:,-self.cols:]
            elif self.type_ == 'seq':
                if self.n_in == 1:
                    return np.squeeze(np.array(X)).T
                if self.n_in > 1:
                    return np.squeeze(np.array(X)).T[-1]
            
            
def supervised_mlp(in_steps, in_features, out_features, 
                   pred_activation='relu'):
    """Return fully supervised Keras multi-layer perceptron.
    
    Define one-headed MLP with `in_steps` times `in_features` 
    inputs and one hidden layer with `out_features` predictions.
    
    Use the Mean Absolute Error (MAE) loss function and 
    the efficient Adam version of stochastic gradient descent.
    """
    
    # Input layer
    input_ = Input(shape=(in_steps * in_features,))
    # Compression layer (optional)
    # compress = Dense(100, activation='relu', 
    #                  use_bias=True)(input_)
    # Prediction layer
    output = Dense(out_features, activation=pred_activation, 
                   # kernel_regularizer=L1L2(l1=0.0001, l2=0.0001), 
                   use_bias=True)(input_)
    model = Model(inputs=input_, outputs=output)
    model.compile(optimizer='adam', loss='mae')
    # model.summary()
    # print(f"Params: {model.count_params()}")
    return model


def sequential_mlp(in_steps, in_features, out_features, 
                   pred_activation='relu'):
    """Return Keras multi-layer perceptron with sequential 
    input data.
    
    Define `in_features` independent multi-headed MLP 
    with `in_steps` neurons in the first hidden layer, 
    concatenate them and return `out_features` predictions.
    
    The input shape will be `in_steps` time steps with 
    `in_features` features.
    
    Use the Mean Absolute Error (MAE) loss function and 
    the efficient Adam version of stochastic gradient descent.
    """
    
    # Input models (one for each feature)
    visibles = []
    denses = []
    for i in range(in_features):
        visibles.append(Input(shape=(in_steps,)))
        denses.append(Dense(in_steps, activation='relu', 
                            # kernel_regularizer=L1(l1=0.0001), 
                            use_bias=True)(visibles[i]))
    # Merge input models
    merge = concatenate(denses)
    output = Dense(out_features, activation=pred_activation, 
                   use_bias=True)(merge)
    model = Model(inputs=visibles, outputs=output)
    model.compile(optimizer='adam', loss='mae')
    # model.summary()
    # print(f"Params: {model.count_params()}")
    return model


if __name__ == '__main__':
    pass