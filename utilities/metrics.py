
# Importing the necessary packages
import tensorflow as tf
import sklearn
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K

def CustomConfusionMatrix(y_true, y_pred):
  """
  This function will compute the confusion matrix, and will return true positive,
  true negative, false positive, and false negative for each class.
      
  Arguments:
      
  y_true --> this is the true labels of our dataset. It should be noted that
              these labels should be one-hot encoded. If they are sparse, then 
              this function will not work. 
  y_pred --> this is the predicted labels of our model. It should be noted that
              these labels should be obtained using softmax activation function. 
              Also, these labels represent the funding and this function will 
              convert them into labels.
      
  Return:
      
  A list --> This function will return a list as follows: 
              [TP0, TP1, TP2, FN0, FN1, FN2, FP0, FP1, FP2, TN0, TN1, TN2]
              where,
              TP0 represent the TRUE POSITIVE for CLASS 0
              TP1 represent the TRUE POSITIVE for CLASS 1
              TP2 represent the TRUE POSITIVE for CLASS 2
              FN0 represent the FALSE NEGATIVE for CLASS 0
              FN1 represent the FALSE NEGATIVE for CLASS 1
              FN2 represent the FALSE NEGATIVE for CLASS 2
              FP0 represent the FALSE POSITIVE for CLASS 0
              FP1 represent the FALSE POSITIVE for CLASS 1
              FP2 represent the FALSE POSITIVE for CLASS 2
              TN0 represent the TRUE NEGATIVE for CLASS 0
              TN1 represent the TRUE NEGATIVE for CLASS 1
              TN2 represent the TRUE NEGATIVE for CLASS 2
  """
  
  def f(y_true, y_pred):
    # It is assumed that there are three classes and both y_true and y_pred are 
    # one-hot encoded
    y_true = tf.argmax(y_true, axis = 1)
    y_pred = tf.argmax(y_pred, axis = 1)
    CM = tf.cast(confusion_matrix(y_true, y_pred), tf.float32)
    # print(y_true, y_pred)
    # TP Class 0, Class 1 and Class 2
    TP0 = CM[0][0]
    TP1 = CM[1][1]
    TP2 = CM[2][2]
    # FN Class 0, Class 1 and Class 2
    FN0 = np.sum(CM[0,1:])
    FN1 = CM[1][0]+CM[1][2]
    FN2 = CM[2][0]+CM[2][1]
    # FP Class 0, Class 1 and Class 2
    FP0 = np.sum(CM[1:,0])
    FP1 = CM[0][1]+CM[2][1]
    FP2 = CM[0][2]+CM[1][2]
    # TN Class 0, Class 1 and Class 2
    TN0 = CM[1][1]+CM[1][2]+CM[2][1]+CM[2][2]
    TN1 = CM[0][0]+CM[0][2]+CM[2][0]+CM[2][2]
    TN2 = CM[0][0]+CM[0][1]+CM[1][0]+CM[1][1]
    # record ={
    #     'TP0': TP0, 'TP1': TP1, 'TP2': TP2,
    #     'FN0': FN0, 'FN1': FN1, 'FN2': FN2,
    #     'TN0': TN0, 'TN1': TN1, 'TN2': TN2,
    #     'FP0': FP0, 'FP1': FP1, 'FP2': FP2
    # }
    return [TP0, TP1, TP2, FN0, FN1, FN2, FP0, FP1, FP2, TN0, TN1, TN2] 
  return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def CustomAccuracy(y_true, y_pred):
  """
    This function will compute the ACCURACY, and will return a scalar as accuracy
    of our model. It should be noted that the returned accuracy would be a macro
    accuracy; however, if you are interested in micro or average accuracy, you can
    amend this function. This function also contains formula to compute the micro
    or average accuracy.
    
    Arguments:
    
    y_true --> this is the true labels of our dataset. It should be noted that
               these labels should be one-hot encoded. If they are sparse, then 
               this function will not work. 
    y_pred --> this is the predicted labels of our model. It should be noted that
               these labels should be obtained using softmax activation function. 
               Also, these labels represent the funding and this function will 
               convert them into labels.
    
    Return:
    
    A Scalar --> accuracy of our model. 
                 It should be noted that the returned accuracy would be a macro
                 accuracy; however, if you are interested in micro or average 
                 accuracy, you can amend this function. This function also 
                 contains formula to compute the micro or average accuracy.
"""

  def f(y_true, y_pred):
    # It is assumed that there are three classes and both y_true and y_pred are 
    # one-hot encoded
    records = CustomConfusionMatrix(y_true, y_pred)
    TP = records[0]+records[1]+records[2]
    FN = records[3]+records[4]+records[5]
    FP = records[6]+records[7]+records[8]
    TN= records[9]+records[10]+records[11]
    
    # Accuracy for Class 0
    Acc0 = (records[0]+records[9])/(records[0]+records[3]+records[6]+records[9]+K.epsilon())
    # Accuracy for Class 1
    Acc1 = (records[1]+records[10])/(records[1]+records[4]+records[7]+records[10]+K.epsilon())
    # Accuracy for Class 2
    Acc2 = (records[2]+records[11])/(records[2]+records[5]+records[8]+records[11]+K.epsilon())
    
    # This is the formula for weighted or micro average accuracy
    accuracy = (TP+ TN) / (TP+TN+FP+FN)

    # for macro average accuracy
    macro_accuracy = (Acc0 + Acc1 + Acc2)/3
    
    return macro_accuracy
  return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def CustomPrecision(y_true, y_pred):
  """
  This function will compute the PRECISION, and will return a scalar as precision
  of our model. It should be noted that the returned precision would be a macro
  precision; however, if you are interested in micro average precision, you can
  amend this function. This function also contains formula to compute the micro
  average precision.
    
  Arguments:
    
  y_true --> this is the true labels of our dataset. It should be noted that
               these labels should be one-hot encoded. If they are sparse, then 
               this function will not work. 
  y_pred --> this is the predicted labels of our model. It should be noted that
               these labels should be obtained using softmax activation function. 
               Also, these labels represent the funding and this function will 
               convert them into labels.
    
  Return:
    
  A Scalar --> precision of our model. 
                 It should be noted that the returned precision would be a macro
                 precision; however, if you are interested in micro average 
                 precision, you can amend this function. This function also 
                 contains formula to compute the micro average precision.
  """
  def f(y_true, y_pred):
    # It is assumed that there are three classes and both y_true and y_pred are 
    # one-hot encoded
    records = CustomConfusionMatrix(y_true, y_pred)
    TP = records[0]+records[1]+records[2]
    FN = records[3]+records[4]+records[5]
    FP = records[6]+records[7]+records[8]
    TN= records[9]+records[10]+records[11]
    
    # Precision for Class 0
    Precision0 = (records[0])/(records[0]+records[6]+K.epsilon())
    # Precision for Class 1
    Precision1 = (records[1])/(records[1]+records[7]+K.epsilon())
    # Precision for Class 2
    Precision2 = (records[2])/(records[2]+records[8]+K.epsilon())
    

    # This is the formula for weighted or micro average precision
    precision = TP / (TP+FP)

    # for macro average precision
    macro_precision = (Precision0 + Precision1 + Precision2)/3
    return macro_precision
    
  return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def CustomRecall(y_true, y_pred):
  """
  This function will compute the RECALL (SENSITIVITY), and will return a scalar as recall
  of our model. It should be noted that the returned recall would be a macro
  recall; however, if you are interested in micro average recall, you can
  amend this function. This function also contains formula to compute the micro
  average recall.
    
  Arguments:
    
  y_true --> this is the true labels of our dataset. It should be noted that
               these labels should be one-hot encoded. If they are sparse, then 
               this function will not work. 
  y_pred --> this is the predicted labels of our model. It should be noted that
               these labels should be obtained using softmax activation function. 
               Also, these labels represent the funding and this function will 
               convert them into labels.
    
  Return:
    
  A Scalar --> recall of our model. 
                 It should be noted that the returned recall would be a macro
                 recall; however, if you are interested in micro average 
                 recall, you can amend this function. This function also 
                 contains formula to compute the micro average recall.

  """
  
  def f(y_true, y_pred):
    # It is assumed that there are three classes and both y_true and y_pred are 
    # one-hot encoded
    records = CustomConfusionMatrix(y_true, y_pred)
    TP = records[0]+records[1]+records[2]
    FN = records[3]+records[4]+records[5]
    FP = records[6]+records[7]+records[8]
    TN= records[9]+records[10]+records[11]

    # Recall for Class 0
    Recall0 = (records[0])/(records[0]+records[3]+K.epsilon())
    # Recall for Class 1
    Recall1 = (records[1])/(records[1]+records[4]+K.epsilon())
    # Recall for Class 2
    Recall2 = (records[2])/(records[2]+records[5]+K.epsilon())

    # This is the formula for weighted or micro average recall
    recall = TP / (TP+FN)

    # for macro average recall
    macro_recall = (Recall0 + Recall1 + Recall2)/3
    return macro_recall
  return tf.numpy_function(f, [y_true, y_pred], tf.float32)


def CustomSpecificity(y_true, y_pred):
  """
    This function will compute the SPECIFICITY, and will return a scalar as specificity
    of our model. It should be noted that the returned specificity would be a macro
    specificity; however, if you are interested in micro average specificity, you can
    amend this function. This function also contains formula to compute the micro
    average specificity.
    
    Arguments:
    
    y_true --> this is the true labels of our dataset. It should be noted that
               these labels should be one-hot encoded. If they are sparse, then 
               this function will not work. 
    y_pred --> this is the predicted labels of our model. It should be noted that
               these labels should be obtained using softmax activation function. 
               Also, these labels represent the funding and this function will 
               convert them into labels.
    
    Return:
    
    A Scalar --> specificity of our model. 
                 It should be noted that the returned specificity would be a macro
                 specificity; however, if you are interested in micro average 
                 specificity, you can amend this function. This function also 
                 contains formula to compute the micro average specificity.
    """
  
  def f(y_true, y_pred):

    # It is assumed that there are three classes and both y_true and y_pred are 
    # one-hot encoded
    records = CustomConfusionMatrix(y_true, y_pred)
    TP = records[0]+records[1]+records[2]
    FN = records[3]+records[4]+records[5]
    FP = records[6]+records[7]+records[8]
    TN= records[9]+records[10]+records[11]

    # Specificity for Class 0
    Specificity0 = (records[9])/(records[9]+records[6]+K.epsilon())
    # Specificity for Class 1
    Specificity1 = (records[10])/(records[10]+records[7]+K.epsilon())
    # Specificity for Class 2
    Specificity2 = (records[11])/(records[11]+records[8]+K.epsilon())
    

    # This is the formula for weighted or micro average specificity
    specificity = TN / (TN+FP)

    # for macro average specifity
    macro_specificity = (Specificity0 + Specificity1 + Specificity2)/3
    
    return macro_specificity
  return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def CustomF1Score(y_true, y_pred):
  """
  This function will compute the F1Score, and will return a scalar as F1-score
  of our model. 
    
  Arguments:
    
  y_true --> this is the true labels of our dataset. It should be noted that
               these labels should be one-hot encoded. If they are sparse, then 
               this function will not work. 
  y_pred --> this is the predicted labels of our model. It should be noted that
               these labels should be obtained using softmax activation function. 
               Also, these labels represent the funding and this function will 
               convert them into labels.
    
  Return:
    
  A Scalar --> F1-score of our model. 
  """
  def f(y_true, y_pred):
    # It is assumed that there are three classes and both y_true and y_pred are 
    # one-hot encoded
    records = CustomConfusionMatrix(y_true, y_pred)
    TP = records[0]+records[1]+records[2]
    FN = records[3]+records[4]+records[5]
    FP = records[6]+records[7]+records[8]
    TN= records[9]+records[10]+records[11]
    
    # Precision for Class 0
    Precision0 = (records[0])/(records[0]+records[6]+K.epsilon())
    # Precision for Class 1
    Precision1 = (records[1])/(records[1]+records[7]+K.epsilon())
    # Precision for Class 2
    Precision2 = (records[2])/(records[2]+records[8]+K.epsilon())
    # Recall for Class 0
    Recall0 = (records[0])/(records[0]+records[3]+K.epsilon())
    # Recall for Class 1
    Recall1 = (records[1])/(records[1]+records[4]+K.epsilon())
    # Recall for Class 2
    Recall2 = (records[2])/(records[2]+records[5]+K.epsilon())

    # F1Score for Class 0
    F1Score0 = (2*Precision0*Recall0)/(Precision0+Recall0+K.epsilon())
    # F1Score for Class 1
    F1Score1 = (2*Precision1*Recall1)/(Precision1+Recall1+K.epsilon())
    # F1Score for Class 2
    F1Score2 = (2*Precision2*Recall2)/(Precision2+Recall2+K.epsilon())
    


    # Macro Averaged F1 Score
    F1Score = (F1Score0 + F1Score1 + F1Score2)/3
    
    return F1Score
  return tf.numpy_function(f, [y_true, y_pred], tf.float32)
