
import os
import matplotlib
matplotlib.use('TKAgg', force=True)
from matplotlib import pyplot as plt


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

def getColorFor(nr):
	if 0 <= nr <= 4: # apple
		return (1.0,0.0,0.0,1.0)
	elif 12 <= nr <= 15:
		return (0.0,0.0,1.0,1.0)
	elif 53 <= nr <= 60:
		return (0.0,1.0,0.0,1.0)
	elif 266 <= nr <= 273:
		return (0.0,1.0,1.0,1.0)
	elif 72 <= nr <= 76:
		return (1.0,0.0,1.0,1.0)
	else:
		grey = 0.3
		return (grey, grey, grey, 1.0)

def printArray(array, name):
	figure = plt.figure()
	fig = figure.add_subplot(1, 1, 1)
	plt.title(str(ele[1] + "/" + ele[0] + " " + ele[2] + ":" + ele[3] + ":" + ele[4] + " " + name))
	x = [e for e in range(len(array[0]))]
	for nr in range(len(array)):
		fig.plot(x, array[nr], color = getColorFor(nr))

path = "../cmake-build-release/2017/"
logFiles = []
for month in os.listdir(path):
	for day in os.listdir(path + month):
		#if "14" in day:
		newPath = path + month + "/" + day
		for hour in os.listdir(newPath):
			if os.path.isfile(newPath + "/" + hour + "/" + "log.txt"):
				hourC = hourConvert(hour)
				t = (newPath + "/" + hour + "/" + "log.txt", convert(month, day, hour), (month, day, str(hourC[0]), str(hourC[1]), str(hourC[2])))
				logFiles.append(t)

logFiles.sort(key=lambda tup: tup[1])
logFiles.reverse()
amount = int(raw_input("Nr of last log files: "))
counterFig = 0

#logFiles = [("/home/denn_ma/workspaceORF/RandomForestWithGPs/RandomForestWithGPs/cmake-build-release/2017/6/12/8:59:53.5/" + "log.txt", convert(month, day, hour), (month, day, str(hourC[0]), str(hourC[1]), str(hourC[2])))]

files = []



for fileN in logFiles:
	counter2 = 0
	lines = open(fileN[0]).read().split("\n")
	if len(lines) > 50:
		for line in lines:
			if "Result:" in line:
				counter2 += 1
			if counter2 > 60:
				files.append(fileN)
				break
	if len(files) > amount + 1:
		break

minA = min(amount, len(files) - 1)

for fileN in files[:minA]:
	points = {}
	found = False
	fakeData = False
	isTest = False
	counter = 0
	currentTestSet = ""
	poolPerformance = [[] for ele in range(300)]
	newSizePool = [[] for ele in range(300)]
	currentSizePool = [[] for ele in range(300)]
	for line in open(fileN[0]).read().split("\n"):
		if "fakeData" in line:
			fakeData = True
		if "Done on test set" in line:
			isTest = False
		if "On test set" in line:
			isTest = True
		if "Perform test for:" in line:
			currentTestSet = line.split(" ")[-1]
			print(line)
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
		if "Result:" in line:
			#if fakeData:
				#print("For fake data r" + line[1:])
			#else:
			if not fakeData:
				found = True
				ele = fileN[2]
				color = ""
				if isTest:
					color = "\033[35m"
				counter += 1
				print(str(counter) + ": " + str(ele[1] + "/" + ele[0] + " " + ele[2] + ":" + ele[3] + ":" + ele[4] + " " + color + line + "\033[0m"))
			if currentTestSet in points:
				points[currentTestSet].append(float(line.split(" ")[-2]))
			else:
				points[currentTestSet] = [float(line.split(" ")[-2])]
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

	sets = ["apple","banana","coffeemug","stapler","flashlight","keyboard"]
	#print(points)
	newLines = [] 
	if "testSet" in points and "testSetExclude" in points and len(points["testSetExclude"]) > 1:
		ele = fileN[2]
		if len(poolPerformance[0]) > 3:
			printArray(poolPerformance, "performance")
			printArray(newSizePool, "new size")
			printArray(currentSizePool, "current size")
		figure = plt.figure()
		fig = figure.add_subplot(1, 1, 1)
		plt.title(str(ele[1] + "/" + ele[0] + " " + ele[2] + ":" + ele[3] + ":" + ele[4]))
		x = [e for e in range(0, len(points["testSet"]))]	
		x2 = [e for e in range(0, len(points["testSetExclude"]))]		
		fig.plot(x, points["testSet"])
		fig.plot(x2, points["testSetExclude"])
		newLines.append("index")
		for i in x:
			newLines.append(str(i))
		newLines[0] += ",testSet,testSetExclude"
		for i in range(0, min(len(x2), len(x))):
			newLines[i+1] += "," + str(points["testSet"][i]) + "," + str(points["testSetExclude"][i])
		for key, point in points.iteritems():
			name = key
			name = name.replace("TestSet", "")
			usedIndex = 0
			if name in sets:
				usedIndex = 15 + sets.index(name) * 10
			else:
				#print(name)
				continue
			newLines[0] += "," + name
			x3 = [e for e in range(usedIndex, usedIndex + len(point))]
			for i in x:
				if usedIndex <= i < usedIndex + len(point) :
					newLines[i+1] += "," + str(point[i - usedIndex]) 
				else:
					newLines[i+1] += ",0"
			fig.plot(x3, point)
	file = open("../cmake-build-release/result.txt", "w")
	file.write("\n".join(newLines))
	file.close()
	if found:
		print("------------")	

plt.show()
	
