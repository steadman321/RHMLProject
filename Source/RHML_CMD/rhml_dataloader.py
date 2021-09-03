################################################################################################
# rhml_dataloader
# load up precanned data : data sets from `UCI` website
################################################################################################
import sys,os
import numpy as np

# Set up python paths
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from RHML.Utils.fileUtils import loadDataCSV

# problem types
CLASSIFICATION="classification" 
REGRESSION="regression"

# data types
NOMINAL="nominal"
NUMERICAL="numerical"

# supported datasets
DATASET_WINE = "wine"
DATASET_WINEQUALITY = "winequality"
DATASET_HOUSING="housing"
DATASET_CARS="cars"
DATASET_CONCRETE="concrete"
SUPPORTED_DATASETS = [DATASET_WINE,DATASET_WINEQUALITY,DATASET_HOUSING,DATASET_CARS,DATASET_CONCRETE]

# dataroot: the local dir expected to be used as the root for finding data files
dataroot = '.'

##################################################################################
# Dataset Config : add a new section in here for new data to be used via RHML_CMD 
# 
# The new section 'key' should be defined in the above SUPPORTED_DATASETS array also
# 
# For a new dataset you need to specify : 
# DATA_FILE         : the location of the csv file , typically in relation to dataroot above
# DATA_LABEL_COLUMN : which column in the data is sued for the target/label value? 0 indicates first col, 1 indicates last col
# DATA_PROBLEM_TYPE : needs to be one of CLASSIFICATION or REGRESSION
# DATA_TYPE         : csv contains numeric or nominal values , default is numerical : only need this if nominal
# DATA_DELIMITER    : delimeter used in csv file , default will be comma : only need this if not using comma
# 
#  Note : assumption is first row of csv is used for feature names
# 
##################################################################################

# data config keys
DATA_FILE="file"
DATA_LABEL_COLUMN="label_column" # 0 indicates first col, 1 indicates last col
DATA_PROBLEM_TYPE="problem_type" # CLASSIFICATION or REGRESSION
DATA_TYPE="data_type"            # default is numerical : only need this if nominal
DATA_DELIMITER="data_delimeter"  # default will be comma : only need this if not using comma

# map for dataset to data file 
dataset_config = {
    'wine':{
        DATA_FILE:dataroot+'/data/wine/wine.data',
        DATA_LABEL_COLUMN:0,
        DATA_PROBLEM_TYPE: CLASSIFICATION
    },
    'winequality':{
        DATA_FILE:dataroot+'/data/winequality/winequality-red.csv',
        DATA_LABEL_COLUMN:1,
        DATA_PROBLEM_TYPE: REGRESSION,
        DATA_DELIMITER:';'
    },
    'housing':{
        DATA_FILE:dataroot+'/data/housing/housing.data',
        DATA_LABEL_COLUMN:1,
        DATA_PROBLEM_TYPE: REGRESSION
    },
    'cars':{
        DATA_FILE:dataroot+'/data/cars/car.data',
        DATA_LABEL_COLUMN:1,
        DATA_PROBLEM_TYPE: CLASSIFICATION,
        DATA_TYPE:NOMINAL
    },
    DATASET_CONCRETE:{
        DATA_FILE:dataroot+'/data/concrete/concrete.data',
        DATA_LABEL_COLUMN:1,
        DATA_PROBLEM_TYPE: REGRESSION
    }
}

# Method to read above config structure. Returns None for anything missing.
# Can pass in a default to use if it ends up being None (optional)
def getDataConfig(dataKey, configKey, default=None):
    data_config = dataset_config.get(dataKey)
    config_value = data_config.get(configKey,None)
    if config_value==None and default is not None:
        return default
    return config_value

################################################################################################
# loadDataSet
# For a given dataset , go load the appropriate dataset
# If any features of the dataset match any values in the featuresToSkip array, these features
# will be dropped from the data returned
################################################################################################
# For a given dataset , go load the appropriate dataset 
def loadDataSet(dataset_code,featuresToSkip):
    # get hold of data csv file location , exit if can not find it
    datasetfile = getDataConfig(dataset_code,DATA_FILE)
    if datasetfile is None:
        sys.exit(f"Could not find the datafile for code: {dataset_code}")

    # check we know what delimiter to use in the csv file , default is comma
    dataset_delimiter = getDataConfig(dataset_code,DATA_DELIMITER,',')

    # check to see if using nominal data
    dataType = getDataConfig(dataset_code,DATA_TYPE,NUMERICAL)
    
    # check we know which col is used for target data 
    labelColIndex = getDataConfig(dataset_code,DATA_LABEL_COLUMN,-1)

    data,labels,feature_names = loadDataCSV(datasetfile,delimiter=dataset_delimiter,labelCol=labelColIndex,numerical=(dataType==NUMERICAL))

    # now need to 'skip' features i.e find the col for that feature and remove it from the data! 
    # first find the index of the cols that we want to skip 
    skip_indexs = []
    upper_features = list(np.char.upper(feature_names))
    if len(featuresToSkip)>0:
        upper_skips = list(np.char.upper(featuresToSkip))
        for skipping in upper_skips:
            try:
                findex = upper_features.index(skipping)
                skip_indexs.append(findex)
            except:
                print(f"Feature to skip is not a known feature {skipping} - picked from {upper_features}")
    
    # now we amend the data to remove the columns we are skipping !
    data=np.delete(data,skip_indexs,axis=1)
    feature_names=np.delete(feature_names,skip_indexs)

    return data,labels,feature_names


