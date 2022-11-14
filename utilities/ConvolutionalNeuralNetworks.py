
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LayerNormalization, BatchNormalization
from tensorflow.keras.layers import Activation, Layer, Dense, Dropout
from tensorflow.keras.layers import MultiHeadAttention, Add
def CNNLayers(inputs, filters = 16, kernel_size = 3, 
             strides = 1, 
             Normalization = 'layer_normalization'):
    """
    This function will create a CNN block (Conv2D --> BN --> Activation)
    
    Arguments:
    inputs --> This is the input of shape (batch_size, rows, cols, channels)
    filters --> This is the input of number of filters used in the convolution layer.
    kernel_size --> the size of the filter. The default value is 3x3
    strides --. A scalar integer. The default value is 1.
    Normalization --> This input is a string, which will decide the type of normalization,
                      you want to use in your model. The default value is 'layer_normalization'.
                      The other two options are 'batch_normalization' and 'batch_renormalization'.
                      
    Return:
    outputs --> This is the output of the block of shape (batch_size, rows, cols, filters)
    """

    x = Conv2D(filters, kernel_size=kernel_size, padding = 'same',
              strides = strides)(inputs)
    if Normalization == 'layer_normalization':
        x = LayerNormalization()(x)
    elif Normalization == 'batch_normalizatin':
        x = BatchNormalization()(x)
    elif Normalization == 'batch_renormalizatin':
        x = BatchNormalization(renorm = True)(x)
    else:
        print("Unknown Type of Input for Normalizatin")
    
    x = Activation(activation='relu')(x)
    return x


def FeatureSelectionBlock(inputs, blocks = 3, filter_bank = [16, 16, 32], kernel_size = 3,
                   strides = 1, Normalization = 'layer_normalization'):
    """
    This function will be used to create the residual block. 
    
    Arguments:
    
    inputs --> This is the input of shape (batch_size, rows, cols, channels)
    blocks -> This will define the number of blocks you want your model to have. The default value is 3.
    filter_bank --> This input contains the values of fitlers to be used in each residual block.
    kernel_size --> the size of the filter. The default value is 3x3
    strides --. A scalar integer. The default value is 1.
    Normalization --> This input is a string, which will decide the type of normalization,
                      you want to use in your model. The default value is 'layer_normalization'.
                      The other two options are 'batch_normalization' and 'batch_renormalization'.
                      
    Return:
    outputs --> This is the output of the block of shape (batch_size, rows, cols, filters)
    """
    x = inputs
    for block in range(blocks):
        for filters in filter_bank:
            x = CNNLayers(inputs=x, filters=filters)
        x = Add()([x, Conv2D(filters = filter_bank[-1], 
                             kernel_size = kernel_size, padding = 'same')(inputs)])
        inputs = x
    return x

def BottleneckBlock(inputs, filter_bank = [16, 16, 32], kernel_size = 3,
                     strides = 2, Normalization = 'layer_normalization'):
    """
    This function will be used to create the transition block. 
    
    Arguments:
    
    inputs --> This is the input of shape (batch_size, rows, cols, channels)
    blocks -> This will define the number of blocks you want your model to have. The default value is 3.
    filter_bank --> This input contains the values of fitlers to be used in each residual block.
    kernel_size --> the size of the filter. The default value is 3x3
    strides --. A scalar integer. The default value is 1.
    Normalization --> This input is a string, which will decide the type of normalization,
                      you want to use in your model. The default value is 'layer_normalization'.
                      The other two options are 'batch_normalization' and 'batch_renormalization'.
                      
    Return:
    outputs --> This is the output of the block of shape (batch_size, rows, cols, filters)
    """
    x = inputs
    for filters in filter_bank[:-1]:
        x = CNNLayers(inputs=x, filters=filters, strides = 1)
    x = CNNLayers(inputs = x, filters = filter_bank[-1], strides = strides)
    x = Add()([x, Conv2D(filters = filter_bank[-1], strides = strides, 
                         kernel_size = kernel_size, padding = 'same')(inputs)])
    inputs = x
    return x

