import random
import numpy as np

###############################################################################################
# RHMLBootstrap:       
# For a given array of data (dataIn) select, at random (with replacement), the same number of 
# data points as in the oringal, from the original data array
################################################################################################
def RHMLBootstrap(dataIn):
    n = len(dataIn)
    
    outdata=[]

    for i in range(n):
        index = random.randint(0,n-1)
        outdata.append(dataIn[index])
    
    return np.array(outdata)

###############################################################################################
# RHMLBootstrap:       
# For a given array of data (dataIn) select, at random (with replacement), the same number of 
# data points as in the oringal, from the original data array
# (This version takes care of returning the new data with its approprate label)
################################################################################################
def RHMLBootstrap(dataIn,labelsIn):
    n = len(dataIn)
    
    outdata=[]
    outlables=[]

    for i in range(n):
        index = random.randint(0,n-1)
        outdata.append(dataIn[index])
        outlables.append(labelsIn[index])
    
    return np.array(outdata),np.array(outlables)

###################################################################################################
# RHMLWeightedBoostrap:       
# For a given array of data (dataIn) select, at random (with replacement), the same number of 
# data points as in the oringal, from the original data array
# 
# This version takes in a set of weights which are used to influence how likely a sample is selected
####################################################################################################
def RHMLWeightedBoostrap(dataIn, labelsIn, weightsIn):
    n = len(dataIn)
    outdata=[]
    outlables=[]
    featureWidth=dataIn.shape[1]
    
    # since we want the same sample and label selected as a unit we fist zip them together
    dataAndLabels = np.append(dataIn,labelsIn,axis=1)

    # use python native random package to select n items at random using the weights
    selections = np.array(random.choices(dataAndLabels,weightsIn,k=n))

    # return as 2 sets of data , so unzip ....
    outdata = selections[:,0:featureWidth]
    outlables = selections[:,featureWidth]

    return outdata,outlables


###################################################################################################
# RHMLBootstrap_withOOB:       
# For a given array of data (dataIn) select, at random (with replacement), the same number of 
# data points as in the oringal, from the original data array
# 
# This version takes also returns an array of indexs corresponding to the data items not included in 
# the boostraped dataset
####################################################################################################
def RHMLBootstrap_withOOB(dataIn,labelsIn):

    # Size of dataset
    n = len(dataIn)
    
    # Init some cache for boostrapped and OOB items
    boostrap_data=[]
    bootstrap_lables=[]

    # track index of data we have used in the boostrap set 
    inbagIndexes = set()

    for i in range(n):
        index = random.randint(0,n-1)
        boostrap_data.append(dataIn[index])
        bootstrap_lables.append(labelsIn[index])
        inbagIndexes.add(index)
    
    # Want to create list of indexes NOT used 
    allIndexs = set(range(n))
    OOBIndexes = allIndexs.difference(inbagIndexes)

    return np.array(boostrap_data),np.array(bootstrap_lables),OOBIndexes
