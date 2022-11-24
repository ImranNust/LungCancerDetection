from sklearn.metrics import roc_curve, auc
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics(history_file):
    """
    This function will plot different metrics versus epochs

      IT IS ASSUMED THAT THE 'HISTORY_FILE.CSV' CONTAINS FOLLOWING METRICS

      1.  CustomeAccuracy: This indicates the Accuracy of Training Data
      2.  CustomF1Score: This indicates the F1 Score of Training Data
      3.  CustomPrecision: This indicates the Precision of Training Data
      4.  CustomRecall: This indicates the Recall of Training Data
      5.  CustomSpecificity: This indicates the Specificity of Training Data
      6.  loss: This indicates the loss of Training Data
      7.  val_CustomeAccuracy: This indicates the Accuracy of Validation Data
      8.  val_CustomF1Score: This indicates the F1 Score of Validation Data
      9.  val_CustomPrecision: This indicates the Precision of Validation Data
      10.  val_CustomRecall: This indicates the Recall of Validation Data
      11.  val_CustomSpecificity: This indicates the Specificity of Validation Data
      12.  val_loss: This indicates the loss of Validation Data
      
    """
    history = pd.read_csv(history_file)
    plt.figure(figsize = (14, 14))
    plt.subplot(2, 3, 1)
    plt.plot(history.epoch.tolist(), history.CustomAccuracy.tolist(), color='blue', linestyle = 'dashdot', label='Train')
    plt.plot(history.epoch.tolist(), history.val_CustomAccuracy.tolist(), color='red', linestyle="solid", label='Val')
    plt.xticks(np.arange(0, 31, step=2))  # Set label locations.
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(visible = True, which = 'both', color= 'r', linestyle = 'dotted')
    plt.legend()
    
    plt.subplot(2, 3, 2)
    plt.plot(history.epoch.tolist(), history.CustomF1Score.tolist(), color='blue', linestyle = 'dashdot', label='Train')
    plt.plot(history.epoch.tolist(), history.val_CustomF1Score.tolist(), color='red', linestyle="solid", label='Val')
    plt.xticks(np.arange(0, 31, step=2))  # Set label locations.
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.grid(visible = True, which = 'both', color= 'r', linestyle = 'dotted')
    plt.legend()
    
    plt.subplot(2, 3, 3)
    plt.plot(history.epoch.tolist(), history.CustomPrecision.tolist(), color='blue', linestyle = 'dashdot', label='Train')
    plt.plot(history.epoch.tolist(), history.val_CustomPrecision.tolist(), color='red', linestyle="solid", label='Val')
    plt.xticks(np.arange(0, 31, step=2))  # Set label locations.
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.grid(visible = True, which = 'both', color= 'r', linestyle = 'dotted')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(history.epoch.tolist(), history.CustomRecall.tolist(), color='blue', linestyle = 'dashdot', label='Train')
    plt.plot(history.epoch.tolist(), history.val_CustomRecall.tolist(), color='red', linestyle="solid", label='Val')
    plt.xticks(np.arange(0, 31, step=2))  # Set label locations.
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.grid(visible = True, which = 'both', color= 'r', linestyle = 'dotted')
    plt.legend()
    
    plt.subplot(2, 3, 5)
    plt.plot(history.epoch.tolist(), history.CustomSpecificity.tolist(), color='blue', linestyle = 'dashdot', label='Train')
    plt.plot(history.epoch.tolist(), history.val_CustomSpecificity.tolist(), color='red', linestyle="solid", label='Val')
    plt.xticks(np.arange(0, 31, step=2))  # Set label locations.
    plt.xlabel('Epoch')
    plt.ylabel('CustomSpecificity')
    plt.grid(visible = True, which = 'both', color= 'r', linestyle = 'dotted')
    plt.legend()
    
    
    plt.subplot(2, 3, 6)
    plt.plot(history.epoch.tolist(), history.loss.tolist(), color='blue', linestyle = 'dashdot', label='Train')
    plt.plot(history.epoch.tolist(), history.val_loss.tolist(), color='red', linestyle="solid", label='Val')
    plt.xticks(np.arange(0, 31, step=2))  # Set label locations.
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.grid(visible = True, which = 'both', color= 'r', linestyle = 'dotted')
    plt.legend()
    return

    
def plot_roc(y_test, y_score, n_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    plt.figure()
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "lime",
                    "blue", "deeppink", "black", "tan", "green", "red"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=2,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )

    plt.plot([-1, 1], [-1, 1], "k--", lw=4)
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(visible = True, which = 'both', color= 'r', linestyle = 'dotted')
    plt.title("AUC-ROC Curve")
    plt.legend(loc="best")
    plt.savefig('AUCROCCurve.png', dpi = 900)
    plt.show()
    return
