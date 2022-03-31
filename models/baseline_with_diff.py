import numpy as np
import tensorflow as tf

from keras.layers import Dense
from keras.models import Model
from tensorflow.keras.regularizers import L1L2
from keras.layers.merge import concatenate


class SupervisedDiffMLP(Model):
    """Return fully supervised Keras multi-layer perceptron
    with differencing.
    
    Define one-headed MLP with `in_steps` times `in_features` 
    inputs and one hidden layer with `out_features` predictions.
    
    This is subclass version of the functional model that needs 
    to be compiled and fitted in order to be fully defined!
    
    TODO: Currently work only for in_steps = 2.
    """
    def __init__(self, in_steps, in_features, out_features, 
                 pred_activation='relu', use_bias=True,
                 L1=0.0001, L2=0.0001):
        super().__init__()
        # parameters
        if in_steps != 2:
            raise NotImplementedError
        self.in_steps = in_steps
        self.in_features = in_features
        self.out_features = out_features
        self.pred_activation = pred_activation
        self.use_bias = use_bias
        self.L1 = L1
        self.L2 = L2
        # layers
        self.dense = Dense(self.out_features, 
                            activation=self.pred_activation,
                            kernel_regularizer=L1L2(l1=self.L1, l2=self.L2),
                            use_bias=self.use_bias)

    def call(self, inputs):
        # Train model on the difference between X(t-1) and X(t-2)
        # but return actual abundance by adding X(t-1) at the end
        prev = inputs[:, self.in_features:]
        diff = prev - inputs[:, :self.in_features]
        return self.dense(diff) + prev


class SequentialDiffMLP(Model):
    """Return Keras multi-layer perceptron with sequential 
    input data and differencing.
    
    Define `in_features` independent multi-headed MLP 
    with `in_steps` neurons in the first hidden layer, 
    concatenate them and return `out_features` predictions.
    
    The input shape will be `in_steps` time steps with 
    `in_features` features.
    
    This is subclass version of the functional model that needs 
    to be compiled and fitted in order to be fully defined!
    
    TODO: Currently work only for in_steps = 2.
    """
    def __init__(self, in_steps, in_features, out_features,
                 input_activation='relu', pred_activation='relu', 
                 use_input_bias=True, use_pred_bias=True, 
                 input_L1=0.001, input_L2=0.001,
                 pred_L1=0.0001, pred_L2=0.0001):
        super().__init__()
        # parameters
        if in_steps != 2:
            raise NotImplementedError
        self.in_steps = in_steps
        self.in_features = in_features
        self.out_features = out_features
        self.input_activation = input_activation
        self.pred_activation = pred_activation
        self.use_input_bias = use_input_bias
        self.use_pred_bias = use_pred_bias
        self.input_L1 = input_L1
        self.input_L2 = input_L2
        self.pred_L1 = pred_L1
        self.pred_L2 = pred_L2
        # layers
        self.denses = []
        for i in range(self.in_features):
            self.denses.append(Dense(1, activation=self.input_activation, 
                               kernel_regularizer=L1L2(l1=self.input_L1, 
                                                       l2=self.input_L2),
                               use_bias=self.use_input_bias))
        self.output_ = Dense(self.out_features, 
                             activation=self.pred_activation,
                             kernel_regularizer=L1L2(l1=self.pred_L1, 
                                                     l2=self.pred_L2),
                             use_bias=self.use_pred_bias)
                               
    def call(self, inputs):
        # Train model on the difference between X(t-1) and X(t-2)
        # but return actual abundance by adding X(t-1) at the end
        prev, merge = [], []
        for i in range(self.in_features):
            last = inputs[i][:, 1]
            diff = tf.expand_dims(last - inputs[i][:, 0], 1)
            prev.append(tf.expand_dims(last, 1))
            merge.append(self.denses[i](diff))
        merge = concatenate(merge)
        prev = concatenate(prev)
        return self.output_(merge) + prev


if __name__ == '__main__':
    pass