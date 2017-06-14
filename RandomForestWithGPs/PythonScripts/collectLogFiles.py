#!/usr/bin/python

import os
import shutil
from os.path import expanduser
def createFolderIfNecessary(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)

def copy(month, day, hour, logFile):
	pathToFolder =expanduser("~") + "/workspaceORF/RandomForestWithGPs/RandomForestWithGPs/cmake-build-release/2017/" + month \
				   + "/" + day + "/" + hour
	createFolderIfNecessary(pathToFolder)
	print("cp " + logFile + " " + pathToFolder + "/log.txt")
	shutil.copyfile(logFile, pathToFolder + "/log.txt")
	if not os.path.exists(pathToFolder + "/log.txt"):
		print("Something went wrong")

	shutil.copyfile(logFile.replace("log.txt", "usedInit.json"), pathToFolder + "/usedInit.json")

path = "/home_local/denn_ma/log/2017/"
if os.path.exists(path):
	for month in os.listdir(path):
		for day in os.listdir(path + month):
			newPath = path + month + "/" + day
			for hour in os.listdir(newPath):
				logFile = newPath + "/" + hour + "/" + "log.txt"
				if os.path.isfile(logFile):
					copy(month, day, hour, logFile)