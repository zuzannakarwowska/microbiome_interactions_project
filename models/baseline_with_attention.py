import numpy as np
import keras.backend as K

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Layer
from keras.layers.merge import concatenate
from tensorflow.keras.regularizers import L1L2


class BahdanauAttention(Layer):
    """https://machinelearningmastery.com/adding-a-custom-attention-layer-to-
    recurrent-neural-network-in-keras/"""
    def __init__(self, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.W=self.add_weight(name='attention_weight', 
                               shape=(input_shape[1], input_shape[1], 1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', 
                               shape=(input_shape[1], 1), 
                               initializer='zeros', trainable=True)        
        super(BahdanauAttention, self).build(input_shape)
 
    def call(self, x):
        # Alignment scores. Pass them through tanh function
        if x.shape[2] > 1:
            # take difference
            # (force the model to focus on dynamics)
            x_processed = x[:, :, -1] - x[:, :, -2]
        else:
            # take last timestep
            x_processed = x[:, :, -1]
        e = K.tanh(K.dot(x_processed, self.W) + self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        # alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = alpha * x_processed
        return context


def supervised_attention_mlp(in_steps, in_features, out_features, 
                             pred_activation='relu', use_bias=True,
                             L1=0.0001, L2=0.0001):
    """Return fully supervised Keras multi-layer perceptron
    with attention.
    
    Define attention-based one-headed MLP with `in_steps` times 
    `in_features` inputs and one hidden layer with `out_features`
    predictions.
    
    Use the Mean Absolute Error (MAE) loss function and 
    the efficient Adam version of stochastic gradient descent.
    
    IMPORTANT: use order='F' in prepare_supervised_data() function
    to make each row in `preprocess` layer being associated with
    distinct bacterium!
    """
    # Input layer
    input_ = Input(shape=(in_features * in_steps))
    input_reshaped = Reshape((in_features, in_steps))(input_)
    # Attention layer
    context = BahdanauAttention()(input_reshaped)
    input_context = concatenate([input_, context])
    # Prediction layer
    output = Dense(out_features, activation=pred_activation, 
                   kernel_regularizer=L1L2(l1=L1, l2=L2), 
                   use_bias=use_bias)(input_context)
    model = Model(inputs=input_, outputs=output)
    model.compile(optimizer='adam', loss='mae')
    return model
    
    
class SupervisedAttentionMLP(Model):
    """Return fully supervised Keras multi-layer perceptron
    with attention.
    
    Define attention-based one-headed MLP with `in_steps` times 
    `in_features` inputs and one hidden layer with `out_features`
    predictions.
    
    This is subclass version of the functional model that needs 
    to be compiled and fitted in order to be fully defined!
    
    IMPORTANT: use order='F' in prepare_supervised_data() function
    to make each row in `preprocess` layer being associated with
    distinct bacterium!
    """
    def __init__(self, in_steps, in_features, out_features, 
                 pred_activation='relu', use_bias=True,
                 L1=0.0001, L2=0.0001):
        super().__init__()
        # parameters
        self.in_steps = in_steps
        self.in_features = in_features
        self.out_features = out_features
        self.pred_activation = pred_activation
        self.use_bias = use_bias
        self.L1 = L1
        self.L2 = L2
        # layers
        self.attention = BahdanauAttention()
        self.dense = Dense(self.out_features, 
                            activation=self.pred_activation,
                            kernel_regularizer=L1L2(l1=self.L1, l2=self.L2),
                            use_bias=self.use_bias)

    def call(self, inputs):
        input_reshaped = Reshape((self.in_features, self.in_steps))(inputs)
        context = self.attention(input_reshaped)
        input_context = concatenate([inputs, context])
        return self.dense(input_context)
    
    
if __name__ == '__main__':
    pass