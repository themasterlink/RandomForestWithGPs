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

def doRFGP(data, time):
    file = "actSettings.json"
    if os.path.exists(file):
        os.remove(file)
    with open(file, 'w') as outfile:
        json.dump(data, outfile)
    subprocess.check_call("./RandomForestWithGPs --settingsFile " + file + " --samplingAndTraining " + str(time), shell=True)

os.chdir("../Release/")
data["TotalStorage"]["folderLocReal"] = "../mnistOrg/"

for t in range(0,10):
    #data["Forest"]["Trees"]["height"] = t
    #data["TotalStorage"]["folderTestNr"] = t
    data["TotalStorage"]["stepOverTrainingData"] = t
    time = 1000
    doRFGP(data, time)
    #for i in range(0,10):
    