import numpy as np
from numpy import genfromtxt

###############################################################################################
# loadDataCSV
# Read in data from csv file.
# Assumes data is in a certain format: frist row should be feature names 
# Params        :
#   filename    : the filename of the csv to read
#   delimiter   : default is comma, can specify an alternative one
#   labelCol    : indicates which column is to b used as the target values/labels
#   numerical   : is the data numerical or nominal
###############################################################################################
def loadDataCSV(filename,delimiter=',',labelCol=None,numerical=True):

    # first get feature names : strings : since all strings treat it differntly!
    featureData = genfromtxt(filename, delimiter=delimiter,dtype='unicode')
    feature_names = featureData[0,:]

    # trim off the target
    if labelCol == 0:
        feature_names = feature_names[1:]
    else:
        feature_names = feature_names[0:-1]

    # decide if process data as numerical or nominal
    if numerical:
        csvData = genfromtxt(filename, delimiter=delimiter)
    else:
        csvData = genfromtxt(filename, delimiter=delimiter,dtype='unicode')

    # remove first line 
    csvData= csvData[1:,:]

    # split into label and data - sometimes labels are at the start sometimes at the end
    if labelCol is not None:
        # if labelCol = 0 , first is label, else its the last 
        if labelCol==0:
            labels = csvData[:,0]
            data = csvData[:,1:]
        else:
            labels = csvData[:,-1]
            data = csvData[:,0:-1]
        return data,labels,feature_names
    else:
        return csvData,feature_names
