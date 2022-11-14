
# Importing the necessary Packages
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding, Dropout
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import Activation, Layer, Dense, Dropout


# Writing the function
class PatchExtractor(Layer):
    """
    This is a class for extracting patches from given input.
    
    IT SHOULD BE NOTED THAT THIS CLASS ACCEPTS INPUT AS BATCHES OF IMAGES; 
    THEREFORE, IF YOU HAVE SINGLE IMAGE, ADD AN ADITIONAL DIMENSION TO MAKE IT
    OF SIZE (BATCH_SIZE, ROWS, COLS, CHANNELS) INSTEAD OF (ROWS, COLS, CHANNELS)
    """
    def __init__(self, patch_size, batch_size, input_shape):
        """
        The constructor of class 
        
        Parameters:
            patch_size: an integer
            batch_size: an integer
            input_shape: a tuple of something like that (rows, cols, channels)
        Returns:
            patches:  patches of dimension (BATCHSIZE, NUM_PATCHES, LENGTH_OF_PATCHES)
            
        EXPLANATION:
        Above, batchsize represents the batch size, num_patches represent the 
        number of patches, and length of patches represents the length of each
        patch. Because after extracting patches from all channels and concatenating
        they are converted to arrays.
        """
        super(PatchExtractor, self).__init__()
        self.patch_size = patch_size
        self.length_of_patch = int(patch_size * patch_size * input_shape[2])
        self.batch_size = batch_size
        self.num_patches = int(input_shape[0]/self.patch_size * input_shape[1]/self.patch_size)
        
    def build(self, input_shape):
        self.kernel = tf.zeros(shape = [self.batch_size, self.num_patches,
                                        self.length_of_patch],dtype=tf.dtypes.float32)
        super(PatchExtractor, self).build(input_shape)  # Be sure to call this somewhere! 
            
    def call(self, images):
        # batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images=images,
                                           sizes=[1, self.patch_size, self.patch_size, 1],
                                           strides=[1, self.patch_size, self.patch_size, 1],
                                           rates=[1, 1, 1, 1],
                                           padding="VALID")
        patch_dims = patches.shape[-1]
        # print(patch_dims)
        self.kernel = tf.reshape(patches, [self.batch_size, self.num_patches,
                                               self.length_of_patch])
        return self.kernel



# Writing the function
class PatchEncoder(Layer):
    def __init__(self, patch_size, batch_size, input_shape):
        super(PatchEncoder, self).__init__()
        self.num_patches = int(input_shape[0]/patch_size * input_shape[1]/patch_size)
        self.projection_dim = int(patch_size * patch_size * input_shape[2])
        self.batch_size = batch_size,
        # self.patch_size = patch_size,
        w_init = tf.random_normal_initializer()
        class_token = w_init(shape=(1, self.projection_dim), dtype="float32")
        self.class_token = tf.Variable(initial_value=class_token, trainable=True)
        self.projection = Dense(units=self.projection_dim)
        self.position_embedding = Embedding(input_dim=self.num_patches+1, 
                                            output_dim=self.projection_dim)

    # def build(self, input_shape):
    #   self.encoded = tf.zeros(shape = [self.batch_size, 257, 
    #                                    self.projection_dim],
    #                           dtype=tf.dtypes.float32)
    
    #   super(PatchEncoder, self).build(input_shape)

    def call(self, patch):
        batch = tf.shape(patch)[0]
        # reshape the class token embedins
        class_token = tf.tile(self.class_token, multiples = [batch, 1])
        class_token = tf.reshape(class_token, (batch, 1, self.projection_dim))
        # calculate patches embeddings
        patches_embed = self.projection(patch)
        # print('patches_embed_shape: {}'.format(patches_embed.shape))
        patches_embed = tf.concat([patches_embed, class_token], 1)
        # print('patches_embedding_after_class: {}'.format(patches_embed.shape))
        # calcualte positional embeddings
        positions = tf.range(start=0, limit=self.num_patches+1, delta=1)
        positions_embed = self.position_embedding(positions)
        # add both embeddings
        encoded = patches_embed + positions_embed
        return encoded
    
# Writing the function
class MLP(Layer):
    def __init__(self, hidden_features, out_features, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.dense1 = Dense(hidden_features, activation=tf.nn.gelu)
        self.dense2 = Dense(out_features)
        self.dropout = Dropout(dropout_rate)

    def call(self, x):
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        y = self.dropout(x)
        return y


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

class TransformerEncoder(Layer):
    def __init__(self, patch_size, batch_size, input_shape,
                 num_heads=4, num_blocks=12, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.num_patches = int(input_shape[0]/patch_size * input_shape[1]/patch_size)
        self.projection_dim = int(patch_size * patch_size * input_shape[2])
        self.batch_size = batch_size,
        self.blocks = [Block(self.projection_dim, num_heads, dropout_rate) for _ in range(num_blocks)]
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
        # return y
