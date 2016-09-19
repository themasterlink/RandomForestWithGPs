#!/usr/bin/python

import math
import random
import numpy as np
import operator
from math import hypot
import matplotlib.pyplot as plt


import json
from pprint import pprint

with open("../Settings/init.json") as data_file:
    data = json.load(data_file)

dim = data["Training"]["dim"]

center0X = 0.5
center0Y = 0.5

center1X = -0.5
center1Y = -0.5

def generateData(dim, amountOfPoints):
    text = ""
    for i in range(0,amountOfPoints):
        a = []
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        dist = math.cos(2.0 * math.pi * x)*math.sqrt(-2 * math.log(y)) * 0.15
        angle = random.uniform(0, math.pi * 2)
        if(i % 2 == 0):
            xVal = center0X + math.cos(angle) * dist
            yVal = center0Y + math.sin(angle) * dist
            text += str(xVal) + "," + str(yVal) +  "," + str(1) + "\n"
        else:
            xVal = center1X + math.cos(angle) * dist
            yVal = center1Y + math.sin(angle) * dist
            text += str(xVal) + "," + str(yVal) +  "," + str(0) + "\n"
    return text


train = generateData(dim, int(data["Training"]["amount"] / 2))

center0X = -0.5
center0Y = 0.5

center1X = 0.5
center1Y = -0.5

train += generateData(dim, int(data["Training"]["amount"] / 2))

f = open(data["Training"]["path"], "w")
f.write(train)
f.close()

text = generateData(dim, data["Test"]["amount"])
f = open(data["Test"]["path"], "w")
f.write(text)
f.close()
