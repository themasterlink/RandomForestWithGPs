#!/usr/bin/python

import math
import random
import numpy as np
import operator
from math import hypot
import matplotlib.pyplot as plt

print("Start")


file1 = open("../testData/testInput.txt", "r")
file2 = open("../testData/trainedResult.txt", "r")
txt = file1.read()
txt2 = file2.read()
lines = txt.split("\n")
lines2 = txt2.split("\n")

plt.figure(0)
var = 0

for line in lines2: # input
    if len(line) > 3:
        ele = line.split(",")
        if float(ele[2]) == 1 :
            plt.plot([float(ele[0])], [float(ele[1])], 'rs')
        elif float(ele[2]) == 2 :
            plt.plot([float(ele[0])], [float(ele[1])], 'gs')
        else:
            plt.plot([float(ele[0])], [float(ele[1])], 'bs')

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