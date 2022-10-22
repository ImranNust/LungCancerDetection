
# Importing the necessary Packages
import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, Add
from src.utilities.MLP import MLP

# Writing the function
class Block(Layer):
    def __init__(self, projection_dim, num_heads=4, dropout_rate=0.1):
        super(Block, self).__init__()
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(projection_dim * 2, projection_dim, dropout_rate)

    def call(self, x):
        # Layer normalization 1.
        x1 = self.norm1(x) # encoded_patches
        # print('x1 shape: {}'.format(x1.shape))
        # Create a multi-head attention layer.
        attention_output = self.attn(x1, x1)
        # print('attention_output shape: {}'.format(attention_output.shape))
        # Skip connection 1.
        x2 = Add()([attention_output, x]) #encoded_patches
        # print('x2 shape: {}'.format(x2.shape))
        # Layer normalization 2.
        x3 = self.norm2(x2)
        # MLP.
        x3 = self.mlp(x3)
        # print('x3 shape: {}'.format(x3.shape))
        # Skip connection 2.
        y = Add()([x3, x2])
        # print('y shape: {}'.format(y.shape))
        return y
