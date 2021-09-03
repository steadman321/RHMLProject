################################################################################################
# rhml_configreader
# 
# This will handle everything to do with reading the config files
###############################################################################################
import configparser
import csv 
import sys

from RHML_CMD.rhml_dataloader import SUPPORTED_DATASETS

SECTION_DATASETS   = "DATA"
DATASETS_LIST_KEY  = "dataset"
DATASETS_SKIP_KEY  = "skip_features"

SECTION_MODEL   = "MODELS"
MODEL_LIST_KEY  = "run_models"
MODEL_DTREE = 'dt'
MODEL_RF = 'rf'
MODEL_BAG = 'bag'
MODEL_BOOST = 'boost'
SUPPORTED_MODELS = [MODEL_DTREE,MODEL_RF,MODEL_BAG,MODEL_BOOST]

# Must have for some model types
MODEL_RF_NUM_FEATURES="num_features"
MODEL_NUM_TREES="num_trees"

# optional sections of config
SECTION_DEFAULT = "ALL_MODELS"
SECTION_DT_PARAMS = "DT_PARAMS"

# Grid Search
SECTION_GRIDSEARCH="GRID_SEARCH"

# Multi model 
SECTION_MULTIMODEL="MULTI_MODEL"

# Section for general report based things like name
SECTION_REPORT="REPORT"
REPORT_NAME="name"
REPORT_DESC="description"

# can specifyy random seed to use - though has a default of 101
# This is expected to be set in the ALL_MODELS section
RANDOM_SEED="random_seed"

def findSectionWithSublist(config,section_name,list_name,validation_list):
    sections = config.sections()

    if section_name not in sections:
        return []
        # sys.exit(f"The config is missing a {section_name} section")

    theSection = config[section_name]

    if list_name not in theSection:
        return []
        # sys.exit(f"The config {section_name} section is missing a '{list_name}' key")

    the_list = (theSection[list_name]).lower().split(',')

    if validation_list==None:
        return the_list

    listParser = csv.reader(the_list)
    listResults = []
    for item in listParser:
        if len(item)>0:
            if item[0] in validation_list:
                listResults.append(item)
    return  listResults


def findWhichModelsToRun(config):
    modelsToRun= findSectionWithSublist(config,SECTION_MODEL,MODEL_LIST_KEY,SUPPORTED_MODELS)
    if modelsToRun ==[]:
        sys.exit(f"The config is missing a {SECTION_MODEL} section with a {MODEL_LIST_KEY} key")

    print(f"The following {len(modelsToRun)} supported model(s) will be run : {modelsToRun}")

    return modelsToRun

def findWhichGridModelsToRun(config):
    modelsToRun= findSectionWithSublist(config,SECTION_GRIDSEARCH,MODEL_LIST_KEY,SUPPORTED_MODELS)
    if modelsToRun ==[]:
        sys.exit(f"The config is missing a {SECTION_GRIDSEARCH} section with a {MODEL_LIST_KEY} key")

    print(f"The following {len(modelsToRun)} supported model(s) will be run : {modelsToRun}")

    return modelsToRun

def findWhichMultiModelsToRun(config):
    modelsToRun= findSectionWithSublist(config,SECTION_MULTIMODEL,MODEL_LIST_KEY,SUPPORTED_MODELS)
    if modelsToRun ==[]:
        sys.exit(f"The config is missing a {SECTION_MULTIMODEL} section with a {MODEL_LIST_KEY} key")

    print(f"The following {len(modelsToRun)} supported model(s) will be run : {modelsToRun} - based on {SECTION_MULTIMODEL}")

    return modelsToRun

def findDataToLoad(config):
    dataToLoad = findSectionWithSublist(config,SECTION_DATASETS,DATASETS_LIST_KEY,SUPPORTED_DATASETS)

    if len(dataToLoad)!=1:
        sys.exit(f"The config {SECTION_DATASETS} must contain one supported dataset : {dataToLoad}")

    print(f"The following datasets will be used : {dataToLoad}")

    return dataToLoad

def findFeaturesToSkip(config):
    featuresToSkip = findSectionWithSublist(config,SECTION_DATASETS,DATASETS_SKIP_KEY,None)
    print(f"The following features will be skipped : {featuresToSkip}")
    return featuresToSkip


# Get default and model specific cponfig, convert to numbers to use as kwargs into the model creation process
def getModelKwargs(config,modelSection):

    # start with empty dictionary
    modelKwargs = {}

    # first read in default config items and add to the kwargs list - may not exit
    sections = config.sections()
    if SECTION_DEFAULT in sections:
        defaultSection = config._sections[SECTION_DEFAULT]
        # convert to ints (config is all strings!)
        for key in defaultSection:
            try:
                modelKwargs[key] = int(defaultSection[key])
            except:
                sys.exit(f"Config value conversion error : expecting numerical value. Can not convert this value to an int. Key = {key}, value given : {defaultSection[key]}.")
            
        print(f"kwargs now looks like this : {modelKwargs}")

    # now add specific items for the model of choice : adding will override the defaults!
    if modelSection in sections:
        modelSection = config._sections[modelSection]
        # convert to ints (config is all strings!)
        for key in modelSection:
            modelKwargs[key] = int(modelSection[key])
        print(f"kwargs now looks like this : {modelKwargs}")

    return modelKwargs

def adjustKwargsForGridSearch(config,modelKwargs):
    # find details under the GRID_SEARCH area and override any we have all ready with these! 
    gridSection = config._sections[SECTION_GRIDSEARCH]
    # convert to ints (config is all strings!)
    for key in gridSection:
        if key != MODEL_LIST_KEY:
            entry=gridSection[key]
            modelKwargs[key] = convertToInts(entry)
    return modelKwargs

# convertToInts:
# This converts strings to ints but also copes with lists of ints too from a config file
# which are expected to be surrounded by square braces [1,2,3]
def convertToInts(entry):
    if entry.startswith('['):
        # remove the braces, split by comma, create a list 
        entry = entry.strip('[]')
        entry_list = [int(x) for x in entry.split(',')]
        return entry_list
    else:
        return int(entry)


# Note: this ignores the run_models settings and is just interested in the other (moving) paramters essentially
# Although settings in here can be single and override other settigns used else where too!
def getMultiModelKwargs(config):
    # start with empty dictionary
    modelKwargs = {}
    sections = config.sections()
    if SECTION_MULTIMODEL in sections:
        multiSection = config._sections[SECTION_MULTIMODEL]
        # convert to ints (config is all strings!)
        for key in multiSection:
            if key != MODEL_LIST_KEY:
                entry=multiSection[key]
                int_entry = convertToInts(entry)
                modelKwargs[key] = int_entry
    print(f"kwargs now looks like this  : {modelKwargs}")
    return modelKwargs

# note , returns "" if not fund. 
# Also, only returns strings - no conversion done here
def getConfigValueIfExists(config,section,searchKey):
    sections = config.sections()
    if section in sections:
        theSection = config._sections[section]
        for key in theSection:
            if key ==searchKey:
                return theSection[key]
    return ""


def getTitleFromConfig(config):
    title = getConfigValueIfExists(config,SECTION_REPORT,REPORT_NAME)
    return title

def getDescriptionFromConfig(config):
    desc = getConfigValueIfExists(config,SECTION_REPORT,REPORT_DESC)
    return desc

def haveSection(config,sectionName):
    sections = config.sections()
    if sectionName in sections:
        return True
    else:
        return False
def haveModelsSection(config):
    return haveSection(config,SECTION_MODEL)

def haveGridSection(config):
    return haveSection(config,SECTION_GRIDSEARCH)

def haveMultiSection(config):
    return haveSection(config,SECTION_MULTIMODEL)

def getRandomSeedFromConfig(config):
    seed = getConfigValueIfExists(config,SECTION_DEFAULT,RANDOM_SEED)
    return seed

    