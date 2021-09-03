################################################################################################
# rhml_cmd
# A standalone python command line tool to experiment using the RHML ensemble library
# 
# Note : this is all driven by a config file: this needs to be passed in with the -config setting
# (If no -config is passed in a default config file in the local dir called config.ini will be assumed)
# The purpose of this specific script it to: 
# - handle and understand the cmd line inputs, 
# - check a valid config file is available 
# - kick off the 'runner' to do the heavy lifiting .... 
# 
#  example to run : (assume in RHML_CMD dir): 
#               python ./rhml_cmd.py -config=./config/classification/wine/test1.ini
# 
#  add in : -grid  , if want grid-search report (instead of default) single-split report
#  add in : -multi , if want multi-model report
###############################################################################################
import argparse
import os
import sys
import configparser

import rhml_runner,rhml_configreader

# Set up python paths
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


# check we have a config file to use as a default
def checkHaveLocalConfigFile():
    return checkConfigFileExists('config.ini')

def checkConfigFileExists(filelocation):
    return os.path.isfile(filelocation)

# Read the config file into form usuable by the runner
def getConfigParser(configFileLocation):
    try:
        cfgParser = configparser.ConfigParser()
        cfgParser.read(configFileLocation)
    except Exception as err:
        raise Exception(f"Error Parsing config file - please check format. It needs to contain [sections] and name=value pairs only. The following error was rasied:\n {err}")

    return cfgParser

def parseCmdLine():

    cmdParser = argparse.ArgumentParser(description='RHML Command Line')
    cmdParser.add_argument('-config',help="Pass in the location of a config.ini file to drive the experiments (default will look for config.ini in local dir)")
    cmdParser.add_argument('-grid',action='store_true',help="Select this option if want to run grid-search instead of default single train:test split run")
    cmdParser.add_argument('-multi',action='store_true',help="Select this option if want to run multiple model types for a parameter range")
    args= vars(cmdParser.parse_args())

    # We must have some config file from somewhere, so check we were given one or have a local one
    configFile = args['config']
    if not checkHaveLocalConfigFile() and configFile is None:
        sys.exit(f"A config file is required. This can be a file in the current directory caleld config.ini or you can pass it in using the optional flag -config=<file location>")

    # use default if none given: 
    if configFile is None:
        configFile = 'config.ini'

    # check the config file does actual exist
    if not checkConfigFileExists(configFile):
        sys.exit(f"The provided config file does not exist at: {configFile}")
    else:
        print(f"Using config file found at: {configFile}")

    # Read the provided config file, and convert to info needed for the runner
    runtimeSettings = getConfigParser(configFile)

    # Look for optional run arguments e.g may want to run GridSearch instead of the default single-split run
    runType = rhml_runner.RUN_SINGLE
    doGridSearch = args['grid']
    doMultiModel = args['multi']
    if doGridSearch and doMultiModel:
        sys.exit(f"Choose either to run 'grid' or 'multi' but not both.")
    if doGridSearch:
        runType=rhml_runner.RUN_GRID
    elif doMultiModel:
        runType=rhml_runner.RUN_MULTI

    # If not used a flag for runtype, check config settings to see if it is obvious what to run given the config we have
    have_models_section = rhml_configreader.haveModelsSection(runtimeSettings)
    have_grid_section = rhml_configreader.haveGridSection(runtimeSettings)
    have_multi_section = rhml_configreader.haveMultiSection(runtimeSettings)

    if not have_models_section and not have_grid_section and not have_multi_section:
        sys.exit(f"The provided config file must contain one of [MODELS], [GRID_SEARCH] or ,[MULTI_MODEL]. Not found in : {configFile}")
    
    if not doGridSearch and not doMultiModel:
        if have_models_section:
            runType = rhml_runner.RUN_SINGLE
        elif have_grid_section and not have_multi_section:
            runType=rhml_runner.RUN_GRID
        elif have_multi_section and not have_grid_section:
            runType=rhml_runner.RUN_MULTI

        if have_grid_section and have_multi_section:
            sys.exit(f"The config file is ambiquius - contains both [] and [] sections. Please specify which report to run with optional flags. Config file : {configFile}")
        
    # Kick off the runner ......
    rhml_runner.run(runtimeSettings,runType)

# Main
parseCmdLine()

