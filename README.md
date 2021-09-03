# Install Guide

## Prerequisites
You need to have Python installed. This project was built using Python 3.8.2, but any version 3.8 and above should work. This project was developed on MacOS, using Catalina 10.15.7. The instructions below are for Mac/Linux based systems and have not been tested on Windows

## Installation
The steps below include creating a virtual environment within which to install the project dependencies; although this step is optional, to avoid issues with different versions of Python libraries that are already installed on your system this is highly recommended.

* Copy RHMLProject-main.zip to a local directory
* unzip RHMLProject-main.zip
* cd RHMLProject-main/Source
* python3 -m venv projectenv
* source ./projectenv/bin/activate
* pip3 install -r ./RHML/requirements

## Test Installation 
Once the install is complete (and assuming your virtual environment is still active) you can test that all things are working by running one of the pre-canned experiments

* cd ./RHML_CMD
* python3 ./rhml_cmd.py -config= ./config/classification/wine/test1.ini
* The experiment should result in your default browser opening and showing the created report
