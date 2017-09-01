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
import random

with open("../Settings/init.json") as data_file:
	data = json.load(data_file)

def doRFGP(data):
	file = "actSettings.json"
	if os.path.exists(file):
		os.remove(file)
	with open(file, 'w') as outfile:
		json.dump(data, outfile, indent=4)
	subprocess.check_call("./RandomForestWithGPs --settingsFile " + file, shell=True)

os.chdir("../cmake-build-release/")
#data["TotalStorage"]["folderLocReal"] = "../mnistOrg/"

rands = [random.randint(11, 738938) for e in range(15)]
print("Random nrs: " + str(rands))

modes = ["use exponential without min and max", "use exponential with min and max", "use performance", "use gaussian"]
nr = random.randint(11, 738938) % len(modes)
for seed in rands:
	data["OnlineRandomForest"]["acceptanceMode"] = modes[nr % len(modes)]
	print("Do: " + modes[nr % len(modes)])
	nr += 1
	data["main"]["seed"] = seed
	doRFGP(data)

# data["OnlineRandomForest"]["determineBestLayerAmount"] = "true"
# for heights in [12,16,20,28, 36]:
# 	data["OnlineRandomForest"]["Tree"]["height"] = heights
# 	doRFGP(data)
#
#
# data["OnlineRandomForest"]["determineBestLayerAmount"] = "false"
# data["OnlineRandomForest"]["Tree"]["height"] = 36
#
# # amount of test points in each split
# for splitValues in [25, 50, 100, 200, 500, 1000, 2000]:
# 	data["OnlineRandomForest"]["amountOfPointsCheckedPerSplit"] = splitValues
# 	doRFGP(data)
#
# data["OnlineRandomForest"]["amountOfPointsCheckedPerSplit"] = 100
#
# # use proportional
# data["MinMaxUsedSplits"]["useFixedValuesForMinMaxUsedSplits"] = "false"
# # amount of splits depends on the amount of data
# for splitValues in [0.1, 0.2,0.4,0.5,0.6,0.7]:
# 	data["MinMaxUsedSplits"]["minValueFractionDependsOnDataSize"] = splitValues
# 	data["MinMaxUsedSplits"]["maxValueFractionDependsOnDataSize"] = splitValues + 0.25
# 	doRFGP(data)

