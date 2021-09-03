import numpy as np
from dataclasses import dataclass


###############################################################################################
# Class Name    :       RHMLConfusionMatrix
# Description   :       Classification Confusion Matrix - (and Metrics) 
################################################################################################
class RHMLConfusionMatrix():
    # Construtor : Need to provide set of predicted classes and set of actual classes
    def __init__(self,testPredictions,testActuals):
        uniqueClassesP = np.unique(testPredictions)
        uniqueClassesA = np.unique(testActuals)
        self.allClasses = list(set(uniqueClassesA).union(set(uniqueClassesP)))
        self.numberClasses = len(self.allClasses)
        self.confusionMatrix = np.zeros((self.numberClasses,self.numberClasses))

        for prediction,actual in zip(testPredictions,testActuals):
            pindex = self.allClasses.index(prediction)
            aindex = self.allClasses.index(actual)
            # store the actuals in the rows, predicted along the cols
            self.confusionMatrix[aindex,pindex]=self.confusionMatrix[aindex,pindex]+1

        self.metrics = []
        self._calcConfusionMatrixMetrics()

    ################################################################################################
    # _calcConfusionMatrixMetrics : calculate various metrics based on confusion matrix values
    ###############################################################################################
    def _calcConfusionMatrixMetrics(self):

        # sum each of the rows 
        rowSums = self.confusionMatrix.sum(axis=1)

        # sum each of the cols 
        colSums = self.confusionMatrix.sum(axis=0)

        # sum of the whole matrix 
        totalSum = np.sum(rowSums)

        # loop over all classes need to deal with 
        for index in range(self.numberClasses):

            # first calc the TP for this class (index is the col?)
            # the TP is the matrix at index,index 
            TP = self.confusionMatrix[index,index]

            # next calc the FN for this class. 
            # this is the row(index) sum - TP
            # assume col index gives you the 'actual' or true class
            FN = rowSums[index] - TP

            # next calc the FP for this class 
            # this is the col(index) - TP
            # assume col index gives you the 'predicted' class
            FP = colSums[index] - TP

            # next calc the TN for that class
            # this is the total matrix sum - (TP+FP+FN)
            TN = totalSum - (TP+FP+FN)

            # Accuracy 
            AC = (TP+TN)/totalSum

            # Guard against never having any correct predictions for a given class
            if TP==0.0:
                PR=0.0
                RC=0.0
                F1=0.0
            else:

                # Precision
                PR = TP/(TP+FP)

                # Recall
                RC = TP/(TP+FN)

                # F1
                F1 = 2 * (1 / ( (1/PR) + (1/RC) ))

            metrics = confusionMetrics(TP,FP,FN,TN,AC,PR,RC,F1)
            self.metrics.append(metrics)



    ################################################################################################
    # calcMicroMetrics:
    #   Assuming the metrics for each class hacve already been calculated and cached
    #   This will calc the Micro precision, recall and F1 across all classes 
    #   For the micro calculations we use total TP,FP and FN across all classes
    ################################################################################################
    def calcMicroMetrics(self):

        totalTP = sum(m.TP for m in self.metrics)
        totalFP = sum(m.FP for m in self.metrics)
        totalFN = sum(m.FN for m in self.metrics)

        microPrecision = totalTP/(totalTP+totalFP)
        microRecall = totalTP/(totalTP+totalFN)
        microF1 = 2 * (1 / ( (1/microPrecision) + (1/microRecall) ))

        return microPrecision,microRecall,microF1

    ################################################################################################
    # calcMacroMetrics
    #   Assuming the metrics for each class have already been calculated and cached
    #   This will calc the Macro precision, recall and F1 across all classes 
    #   The macro calculations are based on the average of class based metrics
    ################################################################################################
    def calcMacroMetrics(self):

        numClasses = len(self.metrics)
        
        macroPrecision = (sum(m.PR for m in self.metrics))/numClasses
        macroRecall = (sum(m.RC for m in self.metrics))/numClasses
        macroF1 = (sum(m.F1 for m in self.metrics))/numClasses

        return macroPrecision,macroRecall,macroF1

    ################################################################################################
    # calcWeightedMetrics:
    #   Assuming the metrics for each class have already been calculated and cached
    #   This will calc the weighted precision, recall and F1 across all classes 
    #   The weighted calculations are based on the weighted average of class based metrics
    ################################################################################################
    def calcWeightedMetrics(self):

        # calc the class weights
        # we ca nget these from the count per row (actuals) in the confusion matrix
        classWeights= self.confusionMatrix.sum(axis=1)
        totalWeights=sum(classWeights)

        weightedPrecision=sum(m.PR*w for m,w in zip(self.metrics,classWeights))/totalWeights
        weightedRecal=sum(m.RC*w for m,w in zip(self.metrics,classWeights))/totalWeights
        weightedF1 = sum(m.F1*w for m,w in zip(self.metrics,classWeights))/totalWeights

        return weightedPrecision,weightedRecal,weightedF1

    ################################################################################################
    # getMetricsSummary: 
    #   This reports the high level/derived stats like : precision , recall and f2 scores
    #   The results will be in 'actual' class order one row per class, 
    #   followed by the micro,macro and weighted metrics at the end
    ################################################################################################
    def getMetricsSummary(self):
        # to begin with just want to copy the metrics, but only parts of it
        derivedMetrics = ["PR","RC","F1"]
        summaryMetrics = [[getattr(x,attr) for attr in derivedMetrics]  
           for x in self.metrics ]

        # now add in the higher order (across all classes) metrics as next few rows too 
        summaryMetrics.append(list(self.calcMicroMetrics()))
        summaryMetrics.append(list(self.calcMacroMetrics()))
        summaryMetrics.append(list(self.calcWeightedMetrics()))

        return summaryMetrics

    ################################################################################################
    # some simple getters for various results    
    ################################################################################################
    def getConfusionMatrix(self):
        return self.confusionMatrix
    
    def getAllClasses(self):
        return self.allClasses

    def getMetrics(self):
        return self.metrics
@dataclass
class confusionMetrics:
    """Metrics for a confusion matrix, e.g True Positive, False Positive etc """
    TP: float
    FP: float
    FN: float
    TN: float
    AC: float
    PR: float
    RC: float
    F1: float

