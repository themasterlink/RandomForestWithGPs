
import os
import matplotlib
matplotlib.use('TKAgg', force=True)
from matplotlib import pyplot as plt
import numpy as np
import json

def hourConvert(hour):
	eles = hour.split(":")
	return (int(eles[0]), int(eles[1]), int(float(eles[2])))

def convert(month, day, hour):
	res = 0
	res += int(month) * 60 * 60 * 24 * 31
	res += int(day) * 60 * 60 * 24
	eles = hourConvert(hour)
	res += eles[0] * 60 * 60
	res += eles[1] * 60
	res += eles[2]



path = "../cmake-build-release/2017/"
logFiles = []
for month in os.listdir(path):
	if "8" in month:
		for day in os.listdir(path + month):
			newPath = path + month + "/" + day
			for hour in os.listdir(newPath):
				if os.path.isfile(newPath + "/" + hour + "/" + "log.txt"):
					hourC = hourConvert(hour)
					#if not os.path.isfile(newPath + "/" + hour + "/" + "classPerformance.eps"):
					t = (newPath + "/" + hour + "/", convert(month, day, hour), (month, day, str(hourC[0]), str(hourC[1]), str(hourC[2])))
					logFiles.append(t)
logFiles.sort(key=lambda tup: tup[1])
logFiles.reverse()

#logFiles = [("/home/denn_ma/workspaceORF/RandomForestWithGPs/RandomForestWithGPs/cmake-build-release/2017/6/12/8:59:53.5/" + "log.txt", convert(month, day, hour), (month, day, str(hourC[0]), str(hourC[1]), str(hourC[2])))]

files = []

buildAvg = False

initFile = logFiles[0][0] + "usedInit.json"
print("Compare to: " + initFile)
with open(initFile) as data_file:
	cmp = json.load(data_file)
for fileN in logFiles[1:]:
	log = fileN[0] + "log.txt"
	initFile = fileN[0] + "usedInit.json"
	with open(initFile) as data_file:
		data = json.load(data_file)
		isTheSame = True
		for first in ["OnlineRandomForest", "MinMaxUsedSplits", "TotalStorage"]:
			for ele in data[first]:
				if cmp[first][ele] != data[first][ele]:
					isTheSame = False
		if isTheSame:
			print("Is the same log: " + log)
		else:
			print("Is not the same log: " + log)