#!/usr/bin/python

import math
import random
import numpy as np
import operator
from math import hypot
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import json
from pprint import pprint

with open("../Settings/init.json") as data_file:
    data = json.load(data_file)

def plotPoints(fileName):
    y,x,temp = np.loadtxt(fileName).T #Transposed for easier unpacking
    nrows, ncols = 101, 100
    grid = np.fliplr(temp.reshape((nrows, ncols)).T)

    plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest', cmap=cm.rainbow)

plt.figure(0)

plotPoints("visu.txt")

file1 = open(data["Training"]["path"], "r")
txt = file1.read()
lines = []
lines = txt.split("\n")

var = 0

i = 0
for line in lines: # input
    if len(line) > 3:
        ele = line.split(",")
        if float(ele[2]) == 1 :
            plt.plot([float(ele[0])], [float(ele[1])], 'ro')
        else:
            plt.plot([float(ele[0])], [float(ele[1])], 'bo')

plt.show()
plt.close()