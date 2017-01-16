#!/usr/bin/python

#!/usr/bin/python

import math
import random
import numpy as np
import operator
from math import hypot
import matplotlib.pyplot as plt
import os
import subprocess

import json
from pprint import pprint

with open("../Settings/init.json") as data_file:
    data = json.load(data_file)


os.chdir("../ReleaseLinux/")
data["TotalStorage"]["folderLocReal"] = "../washington/"
for t in [24,28,30]:
    data["Forest"]["height"] = t
    for i in range(0,10):
   	    data["TotalStorage"]["folderTestNr"] = i
	    file = "actSettings.json"
	    if os.path.exists(file):
	       os.remove(file)
	    with open(file, 'w') as outfile:
	        json.dump(data, outfile)
	    subprocess.check_call("./RandomForestWithGPs --settingsFile " + file + " --samplingAndTraining 200", shell=True)
