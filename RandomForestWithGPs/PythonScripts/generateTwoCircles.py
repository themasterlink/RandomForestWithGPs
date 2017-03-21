#!/usr/bin/python

import math
import random
import numpy as np
import operator
from math import hypot
import matplotlib.pyplot as plt


import os
import json
from pprint import pprint

with open("../Settings/init.json") as data_file:
    data = json.load(data_file)

dim = data["Training"]["dim"]
stretch = 2
def generateData(dim, amountOfPoints):
    text = ""
    if dim == 0:
        for i in range(0,amountOfPoints / 2):
            a = []
            x = random.uniform(0, 1)
            y = random.uniform(0, 1)
            dist = math.cos(2.0 * math.pi * x)*math.sqrt(-2 * math.log(y)) * 0.1
            angle = random.uniform(0, math.pi * 2)
            xVal = math.cos(angle) * dist * stretch
            yVal = math.sin(angle) * dist
            text += str(xVal) + " " + str(yVal) + "\n"
    else:
        for i in range(0,amountOfPoints / 2):
            a = []
            x = random.uniform(0, 1)
            y = random.uniform(0, 1)
            dist = math.cos(2.0 * math.pi * x)*math.sqrt(-2 * math.log(y)) * 0.1
            angle = random.uniform(0, math.pi * 2)
            outerAngle = random.uniform(0, math.pi * 2)
            xVal = math.cos(angle) * dist + math.cos(outerAngle) * stretch
            yVal = math.sin(angle) * dist + math.sin(outerAngle) * 0.5
            text += str(xVal) + " " + str(yVal) + "\n"
    return text


train = generateData(dim, int(data["Training"]["amount"]))

ele = 0
for center in [1,2]:
    path = data["Training"]["path"] + str(ele) + "/vectors.txt"
    if not os.path.isdir(data["Training"]["path"] + str(ele) + "/"):
        os.mkdir(data["Training"]["path"] + str(ele) + "/")
    if os.path.exists(data["Training"]["path"] + str(ele) + "/vectors.binary"):
        os.remove(data["Training"]["path"] + str(ele) + "/vectors.binary")
    f = open(path, "w")
    f.write(generateData(ele, int(data["Training"]["amount"])))
    f.close()
    ele += 1


#train += generateData(dim, int(data["Training"]["amount"] / 2))

