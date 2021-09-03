###############################################################################################
# Proximity.py
# Functions to help with proximity matrix calcs
# (Also see Plots.py for plotting these too)
###############################################################################################
import numpy as np

def calcProximity(ensembleModels,observations):
    # this assumes model is already fit
    numberOfObservations = observations.shape[0]
    proximityMatrix = np.zeros((numberOfObservations,numberOfObservations))

    # for each base model in the ensemble .... 
    for bag in ensembleModels:
        # find out what terminal node each observation would end up in .... 
        nodeIndexes = bag.calcNodes(observations)

        # Find set of unique terminal nodes we are working with 
        uniqueTerminalNodes = np.unique(nodeIndexes)

        # Find the index of the samples in each of these nodes - this will feed the proximityMartrix
        for tni in uniqueTerminalNodes:
            # includesSample_indexes will contain index of samples in the same node - record these values in the prox matrix!
            includesSample_indexes = np.where(nodeIndexes==tni)[0]
            proximityMatrix = _updateProximityMatrix(proximityMatrix,includesSample_indexes)

    # the final proximity scores are an average over all models
    proximityMatrix = proximityMatrix/(len(ensembleModels))

    return proximityMatrix

def _updateProximityMatrix(proximityMatrix, terminalNodeIndexs):
    nodesSize = terminalNodeIndexs.shape[0]
    for i in range(nodesSize):
        for j in range(nodesSize):
            if i != j:
                proximityMatrix[terminalNodeIndexs[i],terminalNodeIndexs[j]]=proximityMatrix[terminalNodeIndexs[i],terminalNodeIndexs[j]]+1
    return proximityMatrix