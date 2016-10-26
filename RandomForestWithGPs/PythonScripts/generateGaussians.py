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

def generateData(dim, amountOfPoints, center0X, center0Y, size):
    text = ""
    for i in range(0,amountOfPoints):
        a = []
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        dist = math.cos(2.0 * math.pi * x)*math.sqrt(-2 * math.log(y)) * size
        angle = random.uniform(0, math.pi * 2)
        xVal = center0X + math.cos(angle) * dist
        yVal = center0Y + math.sin(angle) * dist
        text += str(xVal) + " " + str(yVal) + "\n"
    return text


ele = 0
centers = []
for i in range(4):
    for j in range(2):
        centers.append((i,j))

for center in centers:
    path = data["Training"]["path"] + str(ele) + "/vectors.txt"
    if not os.path.isdir(data["Training"]["path"] + str(ele) + "/"):
        os.mkdir(data["Training"]["path"] + str(ele) + "/")
    if os.path.exists(data["Training"]["path"] + str(ele) + "/vectors.binary"):
        os.remove(data["Training"]["path"] + str(ele) + "/vectors.binary")
    f = open(path, "w")
    f.write(generateData(dim, int(data["Training"]["amount"]), center[0], center[1], 0.2))
    f.close()
    ele += 1

