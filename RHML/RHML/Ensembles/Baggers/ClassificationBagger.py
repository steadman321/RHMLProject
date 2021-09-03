##############################################################################################################
# Package Name  :     ClassificationBagger.py
# Description   :     Extends the base RHMLBagger to provide classification specific implementation
# ############################################################################################################
from scipy import stats
from RHML.Ensembles.Baggers.BaseBagger import RHMLBagger
from RHML.DTrees.ConfusionMatrix import RHMLConfusionMatrix
import numpy as np


###############################################################################################
# Class Name    :       RHMLClassificationBagger
# Description   :       Classification Bagging Ensemble Implementation 
################################################################################################
class RHMLClassificationBagger(RHMLBagger):
    
    def __init__(self,baseModel,B,samples,labels,**kwargs):
        super(RHMLClassificationBagger,self).__init__(baseModel,B,samples,labels,**kwargs)

    ###############################################################################################
    # _aggregatePredictions: for classification this is the mode of all the predictions produced 
    # by the individual models in the ensemble
    ############################################################################################### 
    def _aggregatePredictions(self,predictions):
        # return the mode of predictions
        return stats.mode(predictions,axis=None)[0][0]

    ###############################################################################################
    # _calcTestScore : calc accurancy and confusionMatrix
    ############################################################################################### 
    def _calcTestScore(self,testPredictions,testActuals):
        # simple accuracy measure 
        predictedCorrectly = [testPredictions == testActuals]
        accuracy = (np.sum(predictedCorrectly))/len(testActuals)

        # This confusionMatrix will also have a bunch of related metrics inside it
        cMatrix = RHMLConfusionMatrix(testPredictions,testActuals)

        return (accuracy,cMatrix)

    def getModelType(self):
        return "Classification Bagger"
