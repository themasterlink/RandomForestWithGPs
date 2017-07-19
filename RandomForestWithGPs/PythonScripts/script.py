#!/usr/bin/python

import math
import random
import numpy as np
import operator
from math import hypot
import matplotlib.pyplot as plt
import os
import subprocess

import json
from pprint import pprint

with open("../Settings/init.json") as data_file:
    data = json.load(data_file)

def doRFGP(data):
    file = "actSettings.json"
    if os.path.exists(file):
        os.remove(file)
    with open(file, 'w') as outfile:
        json.dump(data, outfile)
    subprocess.check_call("./RandomForestWithGPs --settingsFile " + file, shell=True)

os.chdir("../cmake-build-release/")
#data["TotalStorage"]["folderLocReal"] = "../mnistOrg/"

data["OnlineRandomForest"]["Tree"]["determineBestLayerAmount"] = "true"

for heights in [16,20,28,35]:
    data["OnlineRandomForest"]["Tree"]["height"] = heights
    doRFGP(data)


data["OnlineRandomForest"]["Tree"]["determineBestLayerAmount"] = "false"
data["OnlineRandomForest"]["Tree"]["height"] = 35

# amount of test points in each split
for splitValues in [25, 50, 100, 200, 500, 1000]:
    data["OnlineRandomForest"]["amountOfPointsCheckedPerSplit"] = splitValues
    doRFGP(data)

data["OnlineRandomForest"]["amountOfPointsCheckedPerSplit"] = 100

# use proportional
data["MinMaxUsedSplits"]["useFixedValuesForMinMaxUsedSplits"] = "false"
# amount of splits depends on the amount of data
for splitValues in [0.1, 0.2,0.4,0.5,0.6,0.7]:
	data["MinMaxUsedSplits"]["minValueFractionDependsOnDataSize"] = splitValues
	data["MinMaxUsedSplits"]["maxValueFractionDependsOnDataSize"] = splitValues + 0.25
	doRFGP(data)

