
# Imports
import tensorflow as tf
from src.utilities import Block
from tensorflow.keras.layers import Layer, LayerNormalization
from tensorflow.keras.layers import Dropout

class TransformerEncoder(Layer):
    def __init__(self, projection_dim, num_heads=4, num_blocks=12, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.blocks = [Block.Block(projection_dim, num_heads, dropout_rate) for _ in range(num_blocks)]
        self.norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(0.5)

    def call(self, x):
        # Create a [batch_size, projection_dim] tensor.
        # print('After Transformer')
        for block in self.blocks:
          x = block(x)
          
        # print('x shape {} after iteration {}'.format(x.shape, block))
        x = self.norm(x)
        y = self.dropout(x)
        return y
