
import os
import matplotlib
matplotlib.use('TKAgg', force=True)
from matplotlib import pyplot as plt
import numpy as np
import json
import subprocess

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

	return int(res)

setColorStarts = {"apple": 0, "banana": 12, "coffeemug":53, "stapler": 266, "flashlight": 72, "testSet": 400, "testSetExclude":500}

grey = 0.3

def getColorFor(nr):
	if 0 <= nr <= 4: # apple
		return (1.0,0.0,0.0,1.0)
	elif 12 <= nr <= 15:
		return (0.0,0.3215,0.647,1.0)
	elif 53 <= nr <= 60:
		return (0.0,1.0,0.0,1.0)
	elif 266 <= nr <= 273:
		return (0.0,1.0,1.0,1.0)
	elif 72 <= nr <= 76:
		return (1.0,0.0,1.0,1.0)
	elif 400 == nr:
		return (1.0,0.631,0.207,1.0)
	elif 500 == nr:
		return (0.407,0.117,0.4941,1.0)
	else:
		return (grey, grey, grey, 1.0)

def getTimeFrom(line):
	line = line.lstrip()
	if "h" in line:
		minAndSec = line[line.find(" h") + len(" h"):]
		min = minAndSec[minAndSec.find(",") + 1:minAndSec.find(" min")]
		secFull = minAndSec[minAndSec.find(" min") + len(" min"):]
		sec = secFull[secFull.find(",") + 1:secFull.find(" sec")]
		minEle = 0
		if len(min) > 0:
			minEle = int(min)
		else:
			minEle = 0
		secEle = 0
		if len(sec) > 0:
			secEle = float(sec)
		else:
			secEle = 0
		hour = line[:line.find(" h")]
		return (int(hour) * 60 + minEle), secEle, 0
	elif "min" in line:
		sec = line[line.find(" min") + len(" min"):]
		sec = sec[sec.find(",") + 1:sec.find(" sec")]
		secEle = 0
		if len(sec) > 0:
			secEle = float(sec)
		else:
			secEle = 0
		min = line[:line.find(" min")]
		return int(min), secEle, 0
	elif "milisec" in line:
		milisec = line[:line.find(" ")]
		return 0, 0.0,float(milisec)
	elif "sec" in line:
		sec = line[:line.find(" ")]
		return 0, float(sec),0
	else:
		print("Unknown: " + line)
		return 0,0.0,0.0


def writeResFile(points, folder, fileName = "result"):
	newLines = []
	newLines.append("index")
	sets = ["banana","coffeemug","stapler","flashlight","apple","keyboard"]
	max = 0
	for key, point in points.iteritems():
		if len(point) > max:
			max = len(point)
	for i in range(max):
		newLines.append(str(i))
	for key, point in points.iteritems():
		usedIndex = 0
		name = key
		name = name.replace("TestSet","")
		if name in sets:
			usedIndex = 15 + sets.index(name) * 10
		newLines[0] += "," + name
		for i in range(max):
			if usedIndex <= i < usedIndex + len(point) :
				newLines[i+1] += "," + str(point[i - usedIndex])
			else:
				newLines[i+1] += ","
	file = open(folder + fileName + ".txt", "w")
	file.write("\n".join(newLines))
	file.close()


def printArray(array, name, folder):
	figure = plt.figure()
	fig = figure.add_subplot(1, 1, 1)
	plt.title(str(ele[1] + "/" + ele[0] + " " + ele[2] + ":" + ele[3] + ":" + ele[4] + " " + name))
	x = [e for e in range(len(array[0]))]
	savedNr = []
	if folder == "":
		avgResult = []
		lenVal = len(array[0])
		avgResult.append("index")
		for e in range(lenVal):
			avgResult.append(str(e))
	for nr in xrange(0, len(array)):
		nr2 = len(array) - nr - 1
		if folder == "":
			avgResult[0] += "," + str(nr2)
			for e in range(len(array[nr2])):
				avgResult[e+1] += "," + str(array[nr2][e])
		if getColorFor(nr2)[0] == grey and getColorFor(nr2)[1] == grey and getColorFor(nr2)[2] == grey:
			fig.plot(x, array[nr2], color = getColorFor(nr2))
		else:
			savedNr.append(nr2)
	for nr2 in savedNr:
		fig.plot(x, array[nr2], color = getColorFor(nr2))
	figure.savefig(folder + name + ".eps")
	plt.close()
	if folder == "":
		file = open(name + ".tex","w")
		allText = "\n".join(avgResult)
		file.write(allText)
		file.close

def printSimpleArray(array, folderName, name):
	figure = plt.figure()
	fig = figure.add_subplot(1, 1, 1)
	x = [e for e in range(0, len(array))]
	fig.plot(x, array)
	figure.savefig(folderName + name + ".eps")
	plt.close()

def extractAndAddLine(line, array, triggerWord, currentSet):
	if triggerWord in line:
		newLine = line[:len(line) - 2]  # remove  % at the end
		removeTrigger = newLine[newLine.find(triggerWord) + len(triggerWord):]
		if currentSet in array:
			array[currentSet].append(float(removeTrigger))
		else:
			array[currentSet] = [float(removeTrigger)]

path = "../cmake-build-release/2017/"
logFiles = []
for month in os.listdir(path):
	if "8" in month:
		for day in os.listdir(path + month):
			if int(day) == 28:
				newPath = path + month + "/" + day
				for hour in os.listdir(newPath):
					if os.path.isfile(newPath + "/" + hour + "/" + "log.txt"):
						hourC = hourConvert(hour)
						#if not os.path.isfile(newPath + "/" + hour + "/" + "classPerformance.eps"):
						t = (newPath + "/" + hour + "/log.txt" , convert(month, day, hour), (month, day, str(hourC[0]), str(hourC[1]), str(hourC[2])))
						logFiles.append(t)

logFiles.sort(key=lambda tup: tup[1])
logFiles.reverse()
amount = 100000  # int(raw_input("Nr of last log files: "))
counterFig = 0

#logFiles = [("/home/denn_ma/workspaceORF/RandomForestWithGPs/RandomForestWithGPs/cmake-build-release/2017/6/12/8:59:53.5/" + "log.txt", convert(month, day, hour), (month, day, str(hourC[0]), str(hourC[1]), str(hourC[2])))]

files = []


buildAvg = True

# initFile = "../cmake-build-release/2017/8/12/16:23:0.6/usedInit.json" # logFiles[0][0] + "usedInit.json"
# print("Compare to: " + initFile)
# with open(initFile) as data_file:
# 	cmp = json.load(data_file)
# for fileN in logFiles[1:]:
# 	log = fileN[0] + "log.txt"
# 	initFile = fileN[0] + "usedInit.json"
# 	with open(initFile) as data_file:
# 		data = json.load(data_file)
# 		isTheSame = True
# 		for first in ["OnlineRandomForest", "MinMaxUsedSplits", "TotalStorage"]:
# 			for ele in data[first]:
# 				if cmp[first][ele] != data[first][ele]:
# 					isTheSame = False
# 		if isTheSame:
# 			print("Is the same log: " + log)
# 			newT = (fileN[0] + "log.txt", fileN[1], fileN[2])
# 			files.append(newT)
# 		else:
# 			print("Is not the same log: " + log)

for fileN in logFiles:
	counter2 = 0
	lines = open(fileN[0]).read().split("\n")
	if len(lines) > 50:
		for line in lines:
			if buildAvg:
				if "Quit Application" in line:
					files.append(fileN)
					break
			else:
				files.append(fileN)
				break
	if len(files) > amount + 1:
		break

minA = min(amount, len(files))
# minA = 1
# files.append(logFiles[0])
avgeragePoints = {}
averagePointsCollect = {}
avgeragePoints2 = {}
averageCounter = {}
for fileN in files[:minA]:
	sets = ["banana","coffeemug","stapler","flashlight","apple","keyboard"]
	print(fileN[0])
	points = {}
	oc = {}
	uc = {}
	ocBVS = {}
	ucBVS = {}
	found = False
	fakeData = False
	isTest = False
	currentTestSet = ""
	poolPerformance = [[] for ele in range(300)]
	newSizePool = [[] for ele in range(300)]
	predictionTime = []
	updatingTime = []
	currentSizePool = [[] for ele in range(300)]
	avgBigTime = []
	avgDeepTime = []
	deepAmount = []
	bigAmount = []
	updatedTrees = []
	sds = []
	counterOfTotalSplits = 0
	offsetForSplits = 0
	betterCounter = 0
	worseCounter = 0
	counterForSetsSplits = {}
	for line in open(fileN[0]).read().split("\n"):
		if "fakeData" in line:
			fakeData = True
		if "Done on test set" in line:
			isTest = False
		if "On test set" in line:
			isTest = True
		if "Perform test for:" in line:
			currentTestSet = line.split(" ")[-1]
			# print(line)
		if "Class: " in line and " performance: " in line and " current size: " in line:
			restLine = line[len("Class: "):]
			classStr = restLine[:restLine.find(",")]
			tempStr = (restLine[restLine.find(": ") + 2:])
			performanceStr = tempStr[:tempStr.find(",")]
			tempStr = tempStr[tempStr.find("new size: ") + len("new size: "):]
			newSize = tempStr[:tempStr.find(",")]
			currentSize = int(tempStr[tempStr.find("current size: ") + len("current size: "):])
			poolPerformance[int(classStr)].append(float(performanceStr))
			newSizePool[int(classStr)].append(int(newSize))
			currentSizePool[int(classStr)].append(int(currentSize))
		if "Prediction was done in:" in line:
			restLine = line[line.find("for: ") + len("for: "):]
			restLine = restLine[:restLine.find(",")]
			if int(restLine) > 5000:
				restLine = line[line.find("per: ") + len("per: "):]
				minu, sec, mili = getTimeFrom(restLine)
				res = 0.0
				if mili > 0:
					res += mili / 1000
				else:
					res += minu * 60.0 + sec
				predictionTime.append(res)
		if "Updating finished, took:" in line:
			restLine = line[line.find("Updating finished, took:") + len("Updating finished, took:"):]
			minu, sec, milisec = getTimeFrom(restLine)
			res = 0.0
			if milisec > 0:
				res += milisec / 1000
			else:
				res += minu * 60.0 + sec
			updatingTime.append(res)
		if "Avg Time for Dynamic" in line:
			restLine = line[line.find("train: ") + len("train: "):]
			time = restLine[:restLine.find(",")]
			minu, sec, milisec = getTimeFrom(time)
			avgDeepTime.append(minu + sec + milisec / 1000.0)
			amount = int(restLine[restLine.find("trees: ") + len("trees: "):])
			deepAmount.append(amount)
		if "Avg Time for Big" in line:
			restLine = line[line.find("train: ") + len("train: "):]
			time = restLine[:restLine.find(",")]
			minu, sec, milisec = getTimeFrom(time)
			avgBigTime.append(minu + sec + milisec / 1000.0)
			amount = int(restLine[restLine.find("trees: ") + len("trees: "):])
			bigAmount.append(amount)
		if "Performed new step with better correctness of" in line:
			betterCounter += 1
		if "Performed new step with worse correctness of" in line:
			worseCounter += 1
		if "Get all points for: " in line:
			nameOfSet = line[line.find("Get all points for: ") + len("Get all points for: "):]
			if "actualSet" in nameOfSet:
				counterOfTotalSplits += 1
				if len(counterForSetsSplits) == 0:
					offsetForSplits = counterOfTotalSplits
			for name in sets:
				if name + "Set" in nameOfSet:
					if not name in counterForSetsSplits:
						counterForSetsSplits[name] = 1
					else:
						counterForSetsSplits[name] += 1
		if "Calculated " in line and "updated: " in line:
			rest = line[line.find("updated: ") + len("updated: "):]
			updatedTrees.append(int(rest))
		if "Finished sorting," in line:
			sdStr = line[line.find("sd: ") + len("sd: "):]
			sds.append(float(sdStr))
		extractAndAddLine(line, points, "Result:", currentTestSet)
		extractAndAddLine(line, oc, "Overconf:", currentTestSet)
		extractAndAddLine(line, uc, "Underconf:", currentTestSet)
		extractAndAddLine(line, ocBVS, "Overconf BVS:", currentTestSet)
		extractAndAddLine(line, ucBVS, "Underconf BVS:", currentTestSet)

	maxEle = 0
	for classEles in poolPerformance:
		maxEle = max(maxEle, len(classEles))
	if maxEle > 0:
		for i in range(len(poolPerformance)):
			while maxEle != len(poolPerformance[i]):
				poolPerformance[i].insert(0, 0.0) # add a zero add the start to get the right performance for all classes
			while maxEle != len(newSizePool[i]):
				newSizePool[i].insert(0, 0)
			while maxEle != len(currentSizePool[i]):
				currentSizePool[i].insert(0, 0)

	#print(points)
	folderName = os.path.dirname(fileN[0]) + "/"
	newLines = [] 
	if "testSet" in points and "testSetExclude" in points and len(points["testSetExclude"]) > 1:
		# if points["testSet"][-1] > 89.0:
		# 	print("File: " + fileN[0] + ", has max: " + str(points["testSet"][-1]) + " %, in: " + str(np.mean(predictionTime) * 10000) + ", " + str(np.mean(updatingTime)))
		# continue
		ele = fileN[2]
		if len(poolPerformance[0]) > 3:
			if buildAvg:
				for key, point in {"pool": poolPerformance, "newSize": newSizePool, "currentSize": currentSizePool}.iteritems():
					if key in avgeragePoints2:
						for ele1, ele2 in zip(point, avgeragePoints2[key]):
							ele2 += ele1
						averageCounter[key] += 1
					else:
						avgeragePoints2[key] = np.array(point)
						averageCounter[key] = 1
			printArray(poolPerformance, "singlePerformance", folderName)
			printArray(newSizePool, "new size", folderName)
			printArray(currentSizePool, "current size", folderName)
		printSimpleArray(predictionTime, folderName, "prediction")
		printSimpleArray(sds, folderName, "standartDeviations")
		printSimpleArray(updatingTime, folderName, "updatingTime")
		printSimpleArray(avgDeepTime, folderName, "avgDeepTime")
		printSimpleArray(avgBigTime, folderName, "avgBigTime")
		printSimpleArray(bigAmount, folderName, "avgBigAmount")
		printSimpleArray(deepAmount, folderName, "avgDeepAmount")
		if len(updatedTrees) > 0:
			printSimpleArray(updatedTrees, folderName, "amountOfUpdatedTrees")

		usedFileNames = ["classPerformance", "oc", "uc", "ocBVS", "ucBVS"]
		for i, usedPoints in enumerate([points, oc, uc, ocBVS, ucBVS]):
			figure = plt.figure()
			fig = figure.add_subplot(1, 1, 1)
			plt.title(str(ele[1] + "/" + ele[0] + " " + ele[2] + ":" + ele[3] + ":" + ele[4]))

			x = [e for e in range(0, len(usedPoints["testSet"]))]
			x2 = [e for e in range(0, len(usedPoints["testSetExclude"]))]
			fig.plot(x, usedPoints["testSet"], color = getColorFor(setColorStarts["testSet"]))
			fig.plot(x2, usedPoints["testSetExclude"], color = getColorFor(setColorStarts["testSetExclude"]))

			for key, point in usedPoints.iteritems():
				name = key
				if buildAvg and usedPoints == points:
					if key in avgeragePoints:
						avgeragePoints[key] += np.array(point)
						averageCounter[key] += 1
						averagePointsCollect[key].append(np.array(point))
					else:
						avgeragePoints[key] = np.array(point)
						averageCounter[key] = 1
						averagePointsCollect[key] = []
						averagePointsCollect[key].append(np.array(point))

				name = name.replace("TestSet", "")
				usedIndex = 0
				if name in sets:
					usedIndex = offsetForSplits + sets.index(name) * counterForSetsSplits[name]
				else:
					#print(name)
					continue
				x3 = [e for e in range(usedIndex, usedIndex + len(point))]
				fig.plot(x3, point, color = getColorFor(setColorStarts[name]))

			figure.savefig(folderName + usedFileNames[i] + ".eps")
			plt.close()
			writeResFile(usedPoints, folderName, usedFileNames[i])

	# if found:
	# 	print("------------")

if buildAvg:
	avgResult = []
	figure = plt.figure()
	fig = figure.add_subplot(1, 1, 1)
	lenVal = len(avgeragePoints["testSet"])
	avgResult.append("index")
	for e in range(lenVal):
		avgResult.append(str(e))
	for key, point in avgeragePoints.iteritems():
		print(key, averageCounter[key])
		plt.title("Average over last")
		name = key
		name = name.replace("TestSet", "")
		avgResult[0] += "," + name
		usedIndex = 0
		if name in sets:
			usedIndex = offsetForSplits + sets.index(name) * counterForSetsSplits[name]
		else:
			usedIndex = 0
		x = [e for e in range(usedIndex, usedIndex + len(point))]
		point /= averageCounter[key]
		for e in range(lenVal):
			if e < usedIndex:
				avgResult[e+1] += ","
			else:
				avgResult[e+1] += "," + str(point[e-usedIndex])
		fig.plot(x, point, color = getColorFor(setColorStarts[name]))
	textAvg = "\n".join(avgResult)
	file = open("avgResult.txt", "w")
	file.write(textAvg)
	file.close()
	figure.savefig("avgPerformance.eps")
	stdAvg = {}
	for key, points in averagePointsCollect.iteritems():
		sum = np.zeros(len(points[0]))
		print("For " + key + ":")
		for point in points:
			sum += np.square(avgeragePoints[key] - point)
			print("\t" + str(point[-1]))
		sum /= averageCounter[key]
		np.sqrt(sum)
		stdAvg[key] = sum

	writeResFile(stdAvg, "", "stdResult")
	writeResFile(avgeragePoints, "")

	for key, point in avgeragePoints2.iteritems():
		point /= averageCounter[key]
		list = point.tolist()
		printArray(list, "avg" + key, "")


#plt.show()
	
