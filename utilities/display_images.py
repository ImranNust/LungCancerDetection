
# Imports
import matplotlib.pyplot as plt
import numpy as np

# Writing the function
def display_images(iterator):
    """
    This function will display images.
    
    Argument:
    
    iterator --> The input should be an iterator with shape (batch_size, rows, cols, channels)
    
    Return: This function does not return anything; instead, it displays the images of the given 
            iterator.
    """
    classes = list(iterator.class_indices)
    images, labels = iterator.next()
    plt.figure(figsize = (8,8))
    if np.max(images[0,...]) <= 1:
        for i in range(0, 25):
            plt.subplot(5,5,i+1)
            plt.imshow(images[i,...])
            plt.title(classes[np.argmax(labels[i])])
            plt.axis('off')
    else:
        for i in range(0, 25):
            plt.subplot(5,5,i+1)
            plt.imshow(images[i,...].astype('uint8'))
            plt.title(classes[np.argmax(labels[i])])
            plt.axis('off')          
    plt.tight_layout()
