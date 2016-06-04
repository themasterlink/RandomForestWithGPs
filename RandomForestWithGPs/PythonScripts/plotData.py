#!/usr/bin/python

import math
import random
import numpy as np
import operator
from math import hypot
import matplotlib.pyplot as plt
import matplotlib.cm as cm

print("Start")

def plotPoints(fileName):
    y,x,temp = np.loadtxt(fileName).T #Transposed for easier unpacking
    nrows, ncols = 201, 200
    grid = np.fliplr(temp.reshape((nrows, ncols)).T)

    plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest', cmap=cm.rainbow)

plt.figure(0)

plotPoints("../testData/trainedResult2.txt")



file1 = open("../testData/testInput2.txt", "r")
txt = file1.read()
lines = []
lines = txt.split("\n")
#lines2 = txt2.split("\n")

var = 0

i = 0
for line in lines: # input
    if len(line) > 3:
        ele = line.split(",")
        if float(ele[2]) == 1 :
            plt.plot([float(ele[0])], [float(ele[1])], 'ro')
        else:
            plt.plot([float(ele[0])], [float(ele[1])], 'bo')




#plt.axis([-600,800,-1400,0])
plt.show()
plt.close()