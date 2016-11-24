#!/usr/bin/python

import math
import random
import numpy as np
import operator
from math import hypot, sqrt
import matplotlib.pyplot as plt
import os

import json
from pprint import pprint

with open("../Settings/init.json") as data_file:
    data = json.load(data_file)

dim = data["Training"]["dim"]

def generateData(dim, amountOfPoints, center0X, center0Y):
    text = ""
    size = random.uniform(1, 3)
    for i in range(0,amountOfPoints):
        a = []
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        dist = math.cos(2.0 * math.pi * x)*math.sqrt(-2 * math.log(y)) * size * 0.3
        angle = random.uniform(0, math.pi * 2)
        xVal = center0X + math.cos(angle) * dist
        yVal = center0Y + math.sin(angle) * dist
        text += str(xVal) + " " + str(yVal) + "\n"
    return text


ele = 0
centers = []
i = 0
while(i < 10):
    x = random.uniform(0, 1)
    y = random.uniform(0, 1)
    dist = math.cos(2.0 * math.pi * x)*math.sqrt(-2 * math.log(y)) * 0.15
    angle = random.uniform(0, math.pi * 2)
    xVal = math.cos(angle) * 8.5 * random.uniform(0, 1)
    yVal = math.sin(angle) * 8.5 * random.uniform(0, 1)
    takeIt = True
    for center in centers:
        dist = math.sqrt((center[0] - xVal)**2 + (center[1] - yVal)**2)
        if(dist < 3.5):
            takeIt = False
            break
    if takeIt:
        centers.append((xVal, yVal))
        i += 1

for center in centers:
    path = data["Training"]["path"] + str(ele) + "/vectors.txt"
    if not os.path.isdir(data["Training"]["path"] + str(ele) + "/"):
        os.mkdir(data["Training"]["path"] + str(ele) + "/")
    if os.path.exists(data["Training"]["path"] + str(ele) + "/vectors.binary"):
        os.remove(data["Training"]["path"] + str(ele) + "/vectors.binary")
    f = open(path, "w")
    f.write(generateData(dim, int(data["Training"]["amount"]), center[0], center[1]))
    f.close()
    ele += 1

os.system("DYLD_LIBRARY_PATH=/usr/local/include/boost_1_61_0/stage/lib:DYLD_LIBRARY_PATH ../Release/RandomForestWithGPs --useFakeData --onlyDataView")
    
    

