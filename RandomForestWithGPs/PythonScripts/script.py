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

for x1 in [2000,3000,4000,5000,6000]:
    for t in ["false", "true"]:
        data["OnlineRandomForest"]["Tree"]["performRealOnlineUpdate"] = t
        data["OnlineRandomForest"]["Tree"]["Bagging"]["totalAmountOfDataUsedPerTree"] = x1
        doRFGP(data)

