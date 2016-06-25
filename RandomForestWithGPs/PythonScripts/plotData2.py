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
    x,y,temp = np.loadtxt(fileName).T #Transposed for easier unpacking
    print(temp)
    for i in range(0, len(x)):
        if float(temp[i]) != -1:
            c = [min(1,math.fabs(float(temp[i]))), 0.0, 1-min(1,math.fabs(float(temp[i])))]
            plt.plot([float(x[i])], [float(y[i])], 'o',color = c)
        else:
            plt.plot([float(x[i])], [float(y[i])], 'go')


    #nrows, ncols = 101, 101
    #grid = temp.reshape((nrows, ncols))
    #print(grid)
    #plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
#interpolation='nearest', cmap=cm.rainbow)

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
            plt.plot([float(ele[0])], [float(ele[1])], 'rs')
        else:
            plt.plot([float(ele[0])], [float(ele[1])], 'bs')

plt.show()
plt.close()