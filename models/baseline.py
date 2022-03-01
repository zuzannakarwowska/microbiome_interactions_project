import numpy as np
    
    
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
        
    TODO: doesn't work yet for n_in > 1 or n_out > 1
    """
    
    def __init__(self, type_, n_in=1, n_out=1):
        self.type_ = type_
        self.n_in = n_in
        self.n_out = n_out
 
    def predict(self, X):
        if self.n_in > 1 or self.n_out > 1:
            raise NotImplementedError
        else:
            if self.type_ == 'sup':
                return X
            elif self.type_ == 'seq':
                return np.squeeze(np.array(X)).T