
# Imports
import tensorflow as tf

def generate_iterator(path, augmentation = True, color_mode = 'rgb',
                      batch_size = 32, shuffle = True, target_size = (64, 64),
                      seed = None, interpolation = 'bilinear', rescale = 1/255.0):
    """
    This function will generate the iterator, that will be used for training, validation,
    and testing.
    
    Arguments:
    
    path --> This is the path of the original directory. It is assumed that this string contains
             the complete path; like "D:/Datasets/DiabeticRetinopathy/UWF_Dataset/UWF/train".
    augmentation --> It is a boolean. If True, only two augmentation will be applied otherwise, no.
    olor_mode --> It is either 'rgb' or 'gray'. The default value is 'rgb'
    batch_size = An integer, the default value is 32.
    shuffle --> A boolean, and the default value is True. For validation and testing data it should be False.
    target_size --> A tuple mentioning the size of the input image (rows, cols, channels). The default
                    value is (64, 64).
                    
    seed --> An integer. The default value is None
    interpolation --> A string, the default value is  'nearest'
    rescale --> rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value
                provided (after applying all other transformations)
    
    Return:
    
    iterator --> An iterator
    """
    
    if augmentation:
        Generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = rescale,
                                                                    horizontal_flip = True, 
                                                                    vertical_flip = True)
                                                                    # rotation_range = 5,
                                                                    # zoom_range = 0.02,
                                                                    # shear_range = 0.02,
                                                                    # zoom_range = 0.02
                                                                    # samplewise_center=True, 
                                                                    # samplewise_std_normalization= True)
    else:
      Generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = rescale)
        
    Iterator = Generator.flow_from_directory(directory = path, target_size=target_size,
                                             color_mode='rgb', batch_size=batch_size,
                                             shuffle=shuffle, seed=None, interpolation='bilinear',
                                             class_mode = 'sparse')
    return Iterator
