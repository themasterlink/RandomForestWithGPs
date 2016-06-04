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

def generateData(dim, amountOfPoints):
    text = ""
    for i in range(0,amountOfPoints):
        a = []
        y = 0
        for l in range(0, dim):
            val = random.uniform(-1.0, 1.0)
            a.append(val)
            text += str(val) + ","
        for l in range(0, dim-1):
            y+= a[l] * a[l]
            y -= 0.33
        if(a[dim-1] > y):
            text += str(1) + "\n"
        else:
            text += str(0) + "\n"
    return text


train = generateData(dim, data["Training"]["amount"])

f = open(data["Training"]["path"], "w")
f.write(train)
f.close()

text = generateData(dim, data["Test"]["amount"])
f = open(data["Test"]["path"], "w")
f.write(text)
f.close()
