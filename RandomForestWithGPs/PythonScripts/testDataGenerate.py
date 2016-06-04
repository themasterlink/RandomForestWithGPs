#!/usr/bin/python

import math
import random
import numpy as np
import operator
from math import hypot
import matplotlib.pyplot as plt

print("Start")


train = ""

for i in range(0,1000):
    x = random.uniform(-1.0, 1.0)
    y = random.uniform(-1.0, 1.0)
    if((y + 0.25) > x * x):
        train += str(x) + "," + str(y) + "," + str(+1) + "\n"
    else:
        train += str(x) + "," + str(y) + "," + str(0) + "\n"

f = open("../testData/testInput2.txt", "w")
f.write(train)
f.close()

test1 = ""

for i in range(0,1000):
    x = random.uniform(-1.0, 1.0)
    y = random.uniform(-1.0, 1.0)
    if((y + 0.25) > x * x):
        test1 += str(x) + "," + str(y) + "," + str(+1) + "\n"
    else:
        test1 += str(x) + "," + str(y) + "," + str(0) + "\n"

f = open("../testData/testInput3.txt", 'w')
f.write(test1)
f.close()