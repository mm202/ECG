from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pylab as pylab
import pandas as pd
import numpy as np

class Reports:
  '''
  Generate reports for the model
  '''
  def __init__(self, yTrue, yPred, labels=None):
    self.yTrue = yTrue
    self.yPred = yPred
    self.labels = labels
  
  def confusionMatrix(self, normalize=None):
    return confusion_matrix(self.yTrue, self.yPred, normalize=normalize)



  def plotConfusionMatrix(self, normalize=None, cmap='Blues', values_format='.2%'):
    #print(pylab.rcParams)
    params = {'legend.fontsize': 4,
              'figure.figsize': (4, 4),
              'axes.labelsize': 4,
              'axes.titlesize': 4,
              'xtick.labelsize': 4,
              'ytick.labelsize': 4,
              'font.size': 4.0,
              'figure.dpi': 200,
              'legend.frameon': False}
    pylab.rcParams.update(params)
    if not self.labels: 
      print('Labels are not provided!')
    cm = self.confusionMatrix(normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=sorted(self.labels))
    disp.plot(include_values=True, cmap=cmap, values_format=values_format)



  def classificationReport(self, digits=4):
    report=classification_report(self.yTrue, self.yPred, digits=digits)
    return report

  def metrics(self):
    #arrays with length equal to num classes
    cfm = self.confusionMatrix()
    FP = cfm.sum(axis=0) - np.diag(cfm) 
    FN = cfm.sum(axis=1) - np.diag(cfm)
    TP = np.diag(cfm)
    TN = cfm.sum() - (FP + FN + TP)

    TPR = TP/(TP+FN)*100 # Sensitivity, recall
    TNR = TN/(TN+FP)*100 # Specificity, true negative rate
    PPV = TP/(TP+FP)*100 # Precision, positive predictive value (PPV)
    NPV = TN/(TN+FN)*100  # Negative predictive value
    FPR = FP/(FP+TN)*100  # False positive rate
    FNR = FN/(TP+FN)*100  # False negative rate
    ACC = (TP+TN)/(TP+FP+FN+TN)*100  # Accuracy of each class
    out = {'Class':sorted(self.labels),'(PPV)Precision':PPV,'(Sensitivity)Recall':TPR,'Specificity':TNR,'Accuracy':ACC}
    return out

  def metricsTable(self):
    mt = self.metrics()
    df = pd.DataFrame(mt).round(2)
    return df