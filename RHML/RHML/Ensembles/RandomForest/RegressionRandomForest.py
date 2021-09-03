##############################################################################################################
# Package Name  :     RegressionRandomForest.py
# Description   :     Extends the base RHMLRegressionBagger to provide regression Random Forest implementation
# ############################################################################################################
from scipy import stats
from RHML.Ensembles.Baggers.RegressionBagger import RHMLRegressionBagger
from RHML.DTrees.RegressionTree import RHMLRegressionDecisionTree
import random


###############################################################################################
# Class Name    :       RHMLClassificationBagger
# Description   :       Classification Bagging Ensemble Implementation 
################################################################################################
class RHMLRegressionRandomForest(RHMLRegressionBagger):
    ###############################################################################################
    # Params        :
    #   M           : the number of features to use 
    #                 (randomly select this number from all feature indexes when create base model)
    #   B           : the number of base models to include
    #   samples     : the data to use to fit 
    #   labels      : the actual labels for the provided samples
    #   kwargs      : optional params and hyperparams (e.g include_oob). 
    #                 (Note these will get passed on to any base model created too)
    ################################################################################################
    def __init__(self,M, B,samples,labels,**kwargs):
        # to make this RandomForest, need to pick random set of features based on M 
        # and set up kwargs to contain the settings for this 
        self.M = M
        super(RHMLRegressionBagger,self).__init__(RHMLRegressionDecisionTree,B,samples,labels,**kwargs)

    ###############################################################################################
    # _createBaseModel
    #  NOTE: For Random Forest, before we create the base model we need to select which subset of features we will use 
    ###############################################################################################
    def _createBaseModel(self,bootstrapSamples,boostrapLabels,**kwargs):
        featureList = self._selectRandomFeatures(bootstrapSamples)
        kwargs['split_fearture_indexes']=featureList
        
        return self.baseModel(bootstrapSamples,boostrapLabels,**kwargs)

    ###############################################################################################
    # _selectRandomFeatures
    # Choose which features to use in the next base model 
    ###############################################################################################
    def _selectRandomFeatures(self,samples):
        # get number of cold in samples : this is our max feature index
        maxFeatureIndex = samples.shape[1]

        # NOTE : if the max number is less than the number of features we are looking for , 
        # there is no point selecting randomly - as we'll end up with all of them 
        # so just return None here = will be same as bagging
        requiredNumberOfFeatures = self.M
        if requiredNumberOfFeatures >= maxFeatureIndex:
            return None
        selectedFeatures = random.sample(range(0, maxFeatureIndex), requiredNumberOfFeatures)
        selectedFeatures.sort()
        return selectedFeatures

    
    def getModelType(self):
        return "Regression Random Forest"