##############################################################################################################
# Package Name  :     ClassificationRandomForest.py
# Description   :     Extends the base RHMLClassificationBagger to provide classification Random Forest implementation
# ############################################################################################################
from RHML.Ensembles.Baggers.ClassificationBagger import RHMLClassificationBagger
from RHML.DTrees.ClassificationTree import RHMLClassificationDecisionTree
import random

###############################################################################################
# Class Name    :       RHMLClassificationBagger
# Description   :       Classification Bagging Ensemble Implementation 
################################################################################################
class RHMLClassificationRandomForest(RHMLClassificationBagger):

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
        self.M = M
        super(RHMLClassificationBagger,self).__init__(RHMLClassificationDecisionTree,B,samples,labels,**kwargs)

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
        # get number of columns in samples : this is our max feature index
        maxFeatureIndex = samples.shape[1]

        # select a random set of cols , number to select is set by self.M
        # NOTE : if the max number of features we have (maxFeatureIndex) is less than the number of features we are looking for , 
        # there is no point randomly selecting a subset - as we will end up with all of them 
        # so just return None here instead .... will be same as Bagging!
        requiredNumberOfFeatures = self.M
        if requiredNumberOfFeatures >= maxFeatureIndex:
            return None
        
        selectedFeatures = random.sample(range(0, maxFeatureIndex), requiredNumberOfFeatures)
        selectedFeatures.sort()
        return selectedFeatures

    def getModelType(self):
        return "Classification Random Forest"

  

