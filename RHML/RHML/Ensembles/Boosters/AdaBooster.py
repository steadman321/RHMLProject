##############################################################################################################
# Package Name  :     AdaBooster.py
# Description   :     Provides AdaBoosting functionality for the RHML booster classes
# ############################################################################################################
from RHML.DTrees.RegressionTree import RHMLRegressionDecisionTree
import numpy as np
from dataclasses import dataclass
from typing import List
import math

from RHML.DTrees import ClassificationTree as CTree
from RHML.Utils.Bootstrap import RHMLWeightedBoostrap
from RHML.DTrees.ConfusionMatrix import RHMLConfusionMatrix

###############################################################################################
# Class Name    :       RHMLAdaBooster
# Description   :       Ada Boosting Ensemble Implementation 
################################################################################################
class RHMLAdaBooster():
    
    DEFAULT_MAX_DEPTH = 1 # Default is to use stumps, but this can be changed via kwargs

    ###############################################################################################
    # Params        :
    #   B           : the number of base models to include
    #   samples     : the data to use to fit 
    #   labels      : the actual labels for the provided samples
    #   kwargs      : optional params and hyperparams (e.g include_oob). 
    #                 (Note these will get passed on to any base model created too)
    ################################################################################################
    def __init__(self,B,samples,labels,**kwargs):
        self.B = B
        self.samples = samples
        self.labels = labels

        self.K = len(np.unique(self.labels))

        # we pass on kwargs down to the underlying base learner. We want to, by default, use stumps, so set the max_depth (if not already done so!)
        if not kwargs.get('max_depth'):
            kwargs['max_depth'] = RHMLAdaBooster.DEFAULT_MAX_DEPTH

        # Are we going to do the multi-class version of the algo or not? Look for ada_multi_class set to 1
        self.useMultiClass= (kwargs.get('ada_multi_class',0))==1

        # initialise an empty set of base models
        self.ensembleModels = []
        self.ensembleModelWeights = []

        # AdaBoost Algo Step 1: init weights = 1/ N, where N is number of samples
        self.numberSamples = self.samples.shape[0]
        self.sample_weights = np.ones(self.numberSamples)
        self.sample_weights=self.sample_weights/self.numberSamples

        # Boostrap samples - these will be created each iteration based on learners weights 
        # To begin with we have all the samples (ignoring the weights as all equal!)
        bootstrapSamples = self.samples
        bootstrapLabels = self.labels

        # AdaBoost Algo Step 2 : Loop around for however many models you want (B)
        for b in range(self.B):

            #  AdaBoost Algo Step 2a : fit a classifier : create a RHMLClassificationDecisionTree
            ctree = CTree.RHMLClassificationDecisionTree(bootstrapSamples,bootstrapLabels,**kwargs)
            # ctree.describe()

            # AdaBoost Algo Step 2b : calculate the misclassificastion rate for the model (needs to know which samples are in error, and in particualr, the weights for these)
            predictions = ctree.predict(self.samples)
            correctWrong = predictions==self.labels
            wrongWeights=self.sample_weights[correctWrong==False]
            countCorrects=np.sum(correctWrong)
            countWrongWeights = len(wrongWeights)
            sumAllWeights = np.sum(self.sample_weights)
            sumWrongWeights = np.sum(wrongWeights)
            misclassError = sumWrongWeights/sumAllWeights


            # AdaBoost Algo Step 2c : now calc the stage weight i.e the weight to be used when taking predictions from this learner 
            # To avoid infinity guard agasint missclass being 0 or 1
            smallError = 0.0000000000001
            if misclassError==0:
                misclassError=misclassError+smallError
            if misclassError==1:
                misclassError=misclassError-smallError

            stageWeight = math.log( (1-misclassError) /misclassError)
            
            # alt: only do this if a flag has been set to amend the algo.
            # optional Multi-class adjustment : 
            if self.useMultiClass:
                stageWeight=stageWeight+math.log(self.K +1)

            # cache this learner in the ensembles list, along with the stageWeight
            self._addBaseModel(ctree,stageWeight)
            
            # AdaBoost Algo Step 2d : Amend the weigths for the samples (uses above stage calc)
            self.sample_weights[correctWrong==False] = self.sample_weights[correctWrong==False]*math.exp(stageWeight)

            # AdaBoost Algo Step 2e :now re-norm the weights 
            total = np.sum(self.sample_weights)
            self.sample_weights=self.sample_weights/total

            # want a new boostrap data to use this time : uses new adjusted weights, so those data items not modelled well last time get more emphasis this time
            # this then loops again - since using weights will have diff tree this time!
            bootstrapSamples,bootstrapLabels = RHMLWeightedBoostrap(self.samples,
                                                                    self.labels.reshape((self.labels.shape[0], 1)),
                                                                    self.sample_weights.reshape((self.sample_weights.shape[0], 1)))
    

            

    def _addBaseModel(self,model, modelWeight):
        self.ensembleModels.append(model) 
        self.ensembleModelWeights.append(modelWeight)

    ###############################################################################################
    # _calcTestScore : calc accurancy and confusionMatrix
    ###############################################################################################
    def _calcTestScore(self,predictions,actuls):
        
        predictedCorrectly = [predictions == actuls]
        accuracy = (np.sum(predictedCorrectly))/len(actuls)
        
        # calc confusion matrix here : 
        # This confusionMatrix will also have a bunch of related metrics inside it
        cMatrix = RHMLConfusionMatrix(predictions,actuls)
        return (accuracy,cMatrix)


    ################################################################################################
    # public api
    ################################################################################################

    ###############################################################################################
    # predict : calculate a prediction for a given set of test observations  
    # push each observation down each tree, and aggregate the results of all trees
    ###############################################################################################
    def predict(self,testObservations): 
        results = []
        # uniqueClasses = np.unique(self.labels).astype(np.int64)
        uniqueClasses = np.unique(self.labels)

        # keep track of all class counts per obs, per model we trained
        classCounts = np.zeros((len(uniqueClasses),len(testObservations)))

        # for each tree in the ensemble - get predictions for all the observations at once : 
        modelCount=0
        for b in self.ensembleModels:

            thisModelWeight = self.ensembleModelWeights[modelCount]
            # All predictions for all observations, but just for model 'b'
            bRawPredictions = np.array(b.predict(testObservations))
            
            # add a count for the predicted class, but weighted by the models weight : self.ensembleModelWeights
            for i,c in enumerate(uniqueClasses):
                # Find out where we predicted class c ... end up with array of True False here .... 
                predictionForClass_c = bRawPredictions==c
                classCountRow = classCounts[i,:]
                classCountRow = classCountRow +(predictionForClass_c*thisModelWeight)
                classCounts[i,:]=classCountRow

            modelCount=modelCount+1

        # now choose one prediction per observation = the max count per class i.e max per col in classCounts
        # Get the maximum values of each column i.e. along axis 0
        finalPredictions = np.argmax(classCounts,axis=0)
        
        # FinalPredictions is the index of the correct class - the actual class is in the uniqueClasses array!
        finalPredictedClasses = uniqueClasses[finalPredictions]
    
        return finalPredictedClasses

    ################################################################################################
    # test : 
    # using a fully fitted booster, calculate the predictions for a given set of observations, 
    # and compare to the provided actual labels. Return a test score.
    # 
    # testObservations : an array like structure , columns are features, rows are observations
    # testLabels : an array: one row per observation : actual label 
    #################################################################################################
    def test(self,testObservations,testLabels):

        # First we get predictions for our test observations
        testPredictions = self.predict(testObservations)

        # now we need to compare our original testObservations and our predictions 
        score = self._calcTestScore(testPredictions,testLabels)

        return adaBoostTestResults(testPredictions,score)

    ################################################################################################
    # calcFeatureImportance : 
    # Note: when calling calcFeatureImportance on the underlying model, we need to turn off normalisation
    # as that is done at this level across all models in the ensemble. 
    ################################################################################################ 
    def calcFeatureImportance(self):
        # need to calc feature importance for each base learner , then get the mean of those
        featureImportance = np.mean([base.calcFeatureImportance(False) for base in self.ensembleModels],axis=0)
        totalGain = np.sum(featureImportance)
        featureImportance=featureImportance/totalGain
        return featureImportance

    def getSummary(self):
        return []

    def getModelType(self):
        return "AdaBooster"

@dataclass
class adaBoostTestResults:
    """Results from running test(), returns predictions and overall score"""
    predictions: List[float]
    score: float
    


