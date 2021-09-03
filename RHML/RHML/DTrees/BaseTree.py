###############################################################################################
# Package Name  :     BaseTree.py
# Description   :     Base Descision Tree Implementation and Supporting Functionality
################################################################################################
import numpy as np
from dataclasses import dataclass
import math
from typing import List
from RHML.Utils.partitions import get_leftside_of_two_partitions

###############################################################################################
# Class Name    :       RHMLDecisionTree
# Description   :       Base Descision Tree Implementation 
################################################################################################
class RHMLDecisionTree:
    
    # setup some constants for defaults ... 
    DEFAULT_MIN_SPLIT = 25
    DEFAULT_MAX_DEPTH = 1000
    MAX_CLASSES_FOR_SPLIT = 15 #for nominal columns, only support this number of unique classes

    ###############################################################################################
    # Constructor : Need to provide set of data as a minimum to create a tree. 
    # The base tree owns the data. 
    # This constructor will fit the data to a new tree.
    ###############################################################################################
    def __init__(self,x_train,y_train,**kwargs):

        # some default settings for the stopping policy : 
        # min_split : intention of this one is : min number of data points in a region i.e dont split if less than this  
        self.min_datapoints = kwargs.get('min_split',RHMLDecisionTree.DEFAULT_MIN_SPLIT) 
        # max_depth : intention of this one is : if depth of tree is more than this max, dont split any more
        self.max_depth = kwargs.get('max_depth',RHMLDecisionTree.DEFAULT_MAX_DEPTH)      
        # split_fearture_indexes : intention of this one is : if want only a subset of features for split points, set to indexes to include
        self.split_fearture_indexes = kwargs.get('split_fearture_indexes',None)          
        
        # maintain an index for all tree nodes, root node is index=0
        self.nodeIndex = 0
        self.nodes = []

        self.x_train = np.ndarray.copy(x_train)
        self.y_train = np.ndarray.copy(y_train)
        self.totalNumberSamples = len(self.x_train)

        # setup some cache for featureImportance tracking
        self.featureImportanceTracker = np.zeros(self.numberOfFeaturesInData())

        # Create a root node with indexes for all the current data 
        allIndexes = np.arange(len(self.x_train))

        # calc root impurity - needed for feature importance calcs
        rootGini = self._calcImpurity(self.y_train)
        
        self._createNode(allIndexes,rootGini)
        
        # now grow the tree ... just split the root node, and that will recursivly split the others until the 
        # stopping criteria are met!
        self._splitNode(self.nodes[0])

    ###############################################################################################
    # _createNode : creates a new node and adds to total nodes for the tree
    ###############################################################################################
    def _createNode(self,dataIndexes,impurityValue):
        newnode = _RHMLDecisionTreeNode(self.nodeIndex,dataIndexes,impurityValue)
        self.nodeIndex=self.nodeIndex + 1
        self.nodes.append(newnode)
        # return the index used for the new node
        return self.nodeIndex-1

    ###############################################################################################
    # _splitNode
    # this is recursive - 
    # this will calculate the best feature and threshold to split a nodes data to get the best impurity score
    # this will then create 2 new nodes, one for the left region and one for the right region, adding the new nodes to the tree
    # each new node will be passed back into this splitNode method to continue the tree growth until a stopping critiera has been met
    # parentDepth : this starts at 0 for the root node. Each recursive iteration will increase the depth. Stop if depth > max_depth
    ###############################################################################################
    def _splitNode(self,node,parentDepth=0):
        splitDepth = parentDepth+1
        
        # Check if we have met a stopping criteria ..... 
        if ( splitDepth<= self.max_depth and 
             node.getCount() > self.min_datapoints and 
             node.impurity>0):

            # Find the best split point - this is in terms of a feature and a split threshold 
            bestSplit = self._findBestSplit(node)

            # Check to see we found a best split : will not find one if only have one class left in region for example
            if bestSplit.feature>-1:

                # record the best split info into this current node and calc which data should be in which region/node
                node.setSplitPoint(bestSplit.feature, bestSplit.threshold)
                leftIndexes, rightIndexes = self._getSplitData(node)

                # Now create 2 new nodes , one for left data and one for right data
                leftNodeIndex = self._createNode(leftIndexes,bestSplit.impurityLeft)
                rightNodeIndex = self._createNode(rightIndexes,bestSplit.impurityRight)  

                # debug only 
                # leftData = self._getNodeData(self.nodes[leftNodeIndex])
                # rightData = self._getNodeData(self.nodes[rightNodeIndex])
                # print(f"DEBUG: left data size is {len(leftData)}, right size is {len(rightData)}")
                # print(f"DEBUG : the split details are : {bestSplit.feature} {bestSplit.threshold}")

                # save these new 'paths' in the parent node so can traverse the tree later
                node.setSplitPaths(leftNodeIndex,rightNodeIndex)

                #Here we can calc feature importance - or at least a contribution towards it for this feature ....
                self._recordFeatureImportanceContribution(node)   

                # Now split these new nodes as well 
                self._splitNode(self.nodes[leftNodeIndex],splitDepth)
                self._splitNode(self.nodes[rightNodeIndex],splitDepth)

    ##################################################################################################### 
    # _findBestSplit : search for best feature and threshold to split a node to get best impurity score
    # bestFeature = -1 : if we do not find a suitable threshold to split on, this will remain as -1 to indicate no split avaiable here e.g. if only one unique class
    #####################################################################################################
    def _findBestSplit(self,node):

        # initialise all the things we want to find the 'best' values for
        minCost = math.inf
        bestFeature = -1
        bestThreshold = 0
        bestSplitGiniLeft = 0
        bestSplitGiniRight = 0

        #Loop over all known features in the node - so get a count of those and output this ! 
        numberOfFeatures = self.numberOfFeaturesInData()

        fearureIndexs = np.arange(numberOfFeatures)
        # It could be that we ONLY want to use a subset of features. 
        # If so then the optional param: split_feature_indexes should contain the index of features to use
        if self.split_fearture_indexes is not None:
            fearureIndexs=self.split_fearture_indexes

        for featureID in fearureIndexs:
            # get current data in this node 
            nodeDataX,nodeDataY = self._getNodeData(node) 

            # get the data associated with this specific feature
            fdata = nodeDataX[:,featureID]

            # get all thresholds we are interested in testing 
            thresholds = self._calcThresholds(fdata)            
            #For each possible threshold, test the split to see which is best
            for threshold in thresholds:

                # Now split the data we have into 2 regions based on the current feature and threshold 
                if isinstance(threshold,list):
                    trueFalseLeftIndex = np.isin(fdata,np.array(threshold))
                else:
                    trueFalseLeftIndex = fdata < threshold

                leftRegionClasses = nodeDataY[trueFalseLeftIndex]
                rightRegionClasses = nodeDataY[(trueFalseLeftIndex==False)]

                # now calc the Impurity measure for each of the 2 regions 
                # NOTE: variables called 'gini', but actually for regression these will carry RSS not gini for impurity measure
                giniLeft = self._calcImpurity(leftRegionClasses)
                giniRight = self._calcImpurity(rightRegionClasses)

                # now calc the overall cost of this split (looking for lowest cost)
                splitCost = self._calcSplitCost(splitSummary(leftRegionClasses.shape[0],rightRegionClasses.shape[0],giniLeft,giniRight))

                if splitCost<minCost:
                    minCost = splitCost
                    bestFeature = featureID
                    bestThreshold = threshold
                    bestSplitGiniLeft = giniLeft
                    bestSplitGiniRight = giniRight

        # return best feature index and threshold
        return splitResults(bestFeature,bestThreshold,bestSplitGiniLeft,bestSplitGiniRight)

    #####################################################################################################
    # _calcThresholds: work out which thresholds to test for when looking for a node split point. 
    # This depends on the type of data i.e numerical or nominal
    #####################################################################################################
    def _calcThresholds(self,featureData):
        thresholds = []

        if isinstance(featureData[0],str):
            thresholds = self._calcNominalThresholds(featureData)
        else:
            thresholds = self._calcNumericThresholds(featureData)

        return thresholds
    
    def _calcNumericThresholds(self,featureData):
        thresholds = []
        fdata_sorted = np.sort(featureData)
        numberOfPairs = fdata_sorted.shape[0] - 1
        for dindex in range(numberOfPairs):
            # if same value do not try and split between!The mean would be the value itself!Results in empty regions
            if fdata_sorted[dindex] == fdata_sorted[dindex+1]:
                continue
            a_threshold = np.mean([[fdata_sorted[dindex]],[fdata_sorted[dindex+1]]])
            thresholds.append(a_threshold)
        return thresholds

    def _calcNominalThresholds(self,featureData):
        thresholds = []

        # Get unique set of classes in the feature data
        uniqueClasses = np.unique(featureData)
        # LIMIT: we only support splitting where number of classes less than MAX_CLASSES_FOR_SPLIT
        if len(uniqueClasses)>self.MAX_CLASSES_FOR_SPLIT:
            raise Exception("Too many unique classes found in nominal preidctor column ")

        # create set of thresholds for left region 
        thresholds = get_leftside_of_two_partitions(uniqueClasses)
        return thresholds


    #####################################################################################################
    # _getSplitData : 
    # Returns the indexes for the data on the left hand side/right hand side of a split point
    #####################################################################################################
    def _getSplitData(self,node):
        # Get the actual feature col data
        nodeDataX,nodeDataY = self._getNodeData(node) 
        nodeDataX_featureCol = nodeDataX[:,node.getSplitFeatureIndex()]
        nodeDataX_dataIndexes =node.getDataIndexs()

        # the way we split depends on if numeric or not .... 
        theThreshold = node.getSplitThreshold()
        if isinstance(theThreshold,list):
            trueFalseLeftIndex = np.isin(nodeDataX_featureCol,np.array(theThreshold))
        else:
            trueFalseLeftIndex = nodeDataX_featureCol < theThreshold

        leftDatapointIndexes = nodeDataX_dataIndexes[trueFalseLeftIndex]
        rightDatapointIndexes = nodeDataX_dataIndexes[(trueFalseLeftIndex==False)]
        
        return leftDatapointIndexes,rightDatapointIndexes

    #####################################################################################################
    # _findPredictionNode : 
    # This will walk the tree, for the given observation, and find where in the tree it stops
    #####################################################################################################
    def _findPredictionNode(self, observation,nodeId=0):
        currentNode = self.nodes[nodeId]
        splitFeatureIndex = currentNode.getSplitFeatureIndex()

        # Check if we are at a terminal node 
        if splitFeatureIndex == None:
            return currentNode

        # Not at a terminal node, so walk down the tree
        splitThreshold = currentNode.getSplitThreshold()
        observationValue = observation[splitFeatureIndex]

        # do we go left or right? 
        goLeft= False
        if isinstance(splitThreshold,list):
            # For nomimal thresholds we have a set of 'left' classes
            if observationValue in splitThreshold:
                goLeft = True
        else:
            # For numerical thresholds we go left if the value is < threshold
            if observationValue < splitThreshold:
                goLeft = True
        
        if goLeft:
            return self._findPredictionNode(observation,currentNode.getLeftPath())
        else:
            return self._findPredictionNode(observation,currentNode.getRightPath())

    #####################################################################################################
    # _recordFeatureImportanceContribution :
    # Keep track of how a feature used in a split has contributed to overall impurity changes 
    #####################################################################################################
    def _recordFeatureImportanceContribution(self,node):
        # What is the feature index for the parent node? 
        parentFeatureIndex = node.getSplitFeatureIndex()

        # grab the other nodes involved in the split
        leftNode = self.nodes[node.getLeftPath()]
        rightNode = self.nodes[node.getRightPath()]

        # Find proportion of things in each node (parent and 2 child nodes) 
        parentSampleCount = node.getCount()
        leftSampleCount = leftNode.getCount()
        rightSampleCount = rightNode.getCount()

        parentProportion = parentSampleCount/self.totalNumberSamples
        leftProportion = leftSampleCount/self.totalNumberSamples
        rightProportion = rightSampleCount/self.totalNumberSamples

        # now get the impurity score for each node
        parentImpurity = node.impurity
        leftImpurity = leftNode.impurity
        rightImpurity = rightNode.impurity
        
        # finally calc the impurity gain 
        impurityGain = (parentProportion*parentImpurity)-(leftProportion*leftImpurity)-(rightProportion*rightImpurity)
        
        # cache this values
        self.featureImportanceTracker[parentFeatureIndex]=self.featureImportanceTracker[parentFeatureIndex]+impurityGain


    ###############################################################################################
    # various convenience 'getter' methods
    ###############################################################################################    
    def _getNodeData(self,node):
        return self.x_train[node.getDataIndexs(),:],self.y_train[node.getDataIndexs()]

    def _getNodeLabelData(self,node):
        return self.y_train[node.getDataIndexs()]
    
    def _getNodeSplitFeatureData(self,node):
        return self.x_train[node.getDataIndexs(),node.getSplitFeatureIndex()]

    def _getNodeFeatureData(self,node,featureIndex):
        return self.x_train[node.getDataIndexs(),featureIndex]

    def _getNodeNumberClasses(self,node):
        # only calc this once then persist it in the node metadata
        numClasses = node.getNumberOfClasses()
        if numClasses is None:
            labels = self._getNodeLabelData(node)
            uniqueClasses = np.unique(labels)
            numClasses = uniqueClasses.shape[0]
            node.setNumberOfClasses(numClasses)
            
        return numClasses

    def numberOfFeaturesInData(self):
        return self.x_train.shape[1]

    ###############################################################################################
    # public api
    ###############################################################################################

    ################################################################################################
    # predict : using a fully fitted tree, calculate the predictions for a given set of observations
    # testObservations : an array like structure , columns are features, rows are observations
    #################################################################################################
    def predict(self,testObservations): 
        results = []
        for observation in testObservations:
            # first get the 'prediction node'
            predictionNode = self._findPredictionNode(observation)

            # Fetch or Calc prediction for given observation (based on selected node)
            # first, go see if the prediction is already cached for the node and use that if it is ...
            prediction = predictionNode.getPrediction()
            if prediction is None:
                prediction = self._calcPrediction(predictionNode)
                predictionNode.setPrediction(prediction)
            results.append(prediction)

        return results


    ################################################################################################
    # test : 
    #   using a fully fitted tree, calculate the predictions for a given set of observations, 
    #   and compare to the provided actual labels. Return a test score.
    # 
    # testObservations : an array like structure , columns are features, rows are observations
    # testLabels : an array: one row per observation : actual label 
    ################################################################################################# 
    def test(self,testObservations,testLabels):
        # First we get predictions for our test observations
        testPredictions = self.predict(testObservations)

        # now we need to compare our original testObservations and our predictions 
        score = self._calcTestScore(testPredictions,testLabels)

        result = testResults(predictions=testPredictions,score=score)
        return result

    ################################################################################################
    # calcFeatureImportance : 
    #   use data cached by _recordFeatureImportanceContribution to work out final feature importance
    #   Note: for some ensembles the normalisation step is done at that level, 
    #   in which case this should be called with normalise=False
    ################################################################################################
    def calcFeatureImportance(self,normalise=True):
        totalGain = np.sum(self.featureImportanceTracker)
        if normalise:
            finalGain = self.featureImportanceTracker / totalGain
        else:
            finalGain = self.featureImportanceTracker
        return finalGain

    ################################################################################################
    # describe: 
    #   more for debug, this outputs info on a tree structure ....
    ################################################################################################
    def describe(self):
        if len(self.nodes)>0:
            print("*************************************************************************")
            print(f"Have a tree with nodes .... lets take a look at each node .... ")
            for node in self.nodes:
                node.describe()
            print("*************************************************************************")

    ######################################################################################################
    # getSummary:
    #   Wheras 'describe' is human readable summary, this is aligned to the needs of final reporting methods
    ######################################################################################################
    def getSummary(self):
        treeSummary = []
        if len(self.nodes)>0:
            for node in self.nodes:
                nodeSummary = node.getNodeSummary()
                treeSummary.append(nodeSummary)
        return treeSummary

    def getModelType(self):
        return "Core Decision Tree"


    # Unlike predict, this ONLY returns the index of the node an observation ends up in - useful for proximity testing
    # (In proximity testing we are looking for observations in the sanem node)
    def calcNodes(self,testObservations): 
        nodeIndexes = []
        for observation in testObservations:
            # first get the 'prediction node'
            predictionNode = self._findPredictionNode(observation)
            nodeIndex = predictionNode.getIndex()
            nodeIndexes.append(nodeIndex)

        return nodeIndexes


###############################################################################################
# Class Name    :       _RHMLDecisionTreeNode
# Description   :       Private class, only used by RHMLDecisionTree
#                       Encapsulates a tree node, and 'owns' the data, and the stats for a node 
################################################################################################
class _RHMLDecisionTreeNode:
    def __init__(self,index, dataIndexes,impurity):
        self.index = index
        self.dataIndexes = dataIndexes
        self.impurity = impurity
        self.featureSplitIndex=None
        self.splitThreshold=None
        self.leftPathNode = None
        self.rightPathNode = None
        self.numberOfClasses = None
        self.prediction=None

    def getIndex(self):
        return self.index

    def getCount(self):
        return len(self.dataIndexes)
    
    def getDataIndexs(self):
        return self.dataIndexes

    def setSplitPoint(self,featureIndex, threshold):
        self.featureSplitIndex = featureIndex
        self.splitThreshold = threshold
    
    def setSplitPaths(self,leftNodeIndex,rightNodeIndex):
        self.leftPathNode = leftNodeIndex
        self.rightPathNode = rightNodeIndex

    def getSplitFeatureIndex(self):
        return self.featureSplitIndex
    
    def getSplitThreshold(self):
        return self.splitThreshold

    def getLeftPath(self):
        return self.leftPathNode

    def getRightPath(self):
        return self.rightPathNode
    
    def setNumberOfClasses(self,count):
        self.numberOfClasses = count
    
    def getNumberOfClasses(self):
        return self.numberOfClasses
    
    def setPrediction(self,val):
        self.prediction=val

    def getPrediction(self):
        return self.prediction

    def describe(self):
        summaryLine = "Node index "+str(self.index)+"\t"
        summaryLine += ": datasize :"+str(self.getCount())+"\t"
        if self.featureSplitIndex is None:
            summaryLine += " : No Split "
        else:
            summaryLine += " : Split feature: "+str(self.featureSplitIndex)+"\t"
            summaryLine += " threashold: "+ str(self.splitThreshold)+"\t"
            summaryLine += " leftNode: "+ str(self.leftPathNode)+"\t"
            summaryLine += " rightNode: " +str(self.rightPathNode)

        print(summaryLine)
       
    def getNodeSummary(self):
        nodeIndex = str(self.index)
        nodeSize = str(self.getCount())
        nodeSplit= "Terminal - No Split"
        threshold=""
        leftNode=""
        rightNode = ""
        
        if self.featureSplitIndex is not None:
            nodeSplit = str(self.featureSplitIndex)
            threshold = str(self.splitThreshold)
            leftNode = str(self.leftPathNode)
            rightNode = str(self.rightPathNode)

        return [nodeIndex,nodeSize,nodeSplit,threshold,leftNode,rightNode]

# Simple data classes for complext return values
@dataclass 
class splitResults:
    """Class for keeping track of split results"""
    feature: int
    threshold: float
    impurityLeft: float
    impurityRight: float

@dataclass 
class splitSummary:
    """summary of count values of potential split """
    leftRegionCount: int
    rightRegionCount: int
    impurityLeft: float
    impurityRight:float


@dataclass
class testResults:
    """Results from running test(), returns predictions and overall score"""
    predictions: List[float]
    score: float