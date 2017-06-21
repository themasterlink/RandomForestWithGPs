
import os
import matplotlib
matplotlib.use('TKAgg', force=True)
from matplotlib import pyplot as plt


def hourConvert(hour):
	eles = hour.split(":")
	return (int(eles[0]), int(eles[1]), int(float(eles[2])))

def convert(month, day, hour):
	res = 0.0	
	res += int(month) * 60 * 60 * 24 * 31
	res += int(day) * 60 * 60 * 24
	eles = hourConvert(hour)
	res += eles[0] * 60 * 60 
	res += eles[1] * 60 
	res += eles[2]

	return int(res)

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
amount = int(raw_input("Nr of last log files: "))
counterFig = 0

#logFiles = [("/home/denn_ma/workspaceORF/RandomForestWithGPs/RandomForestWithGPs/cmake-build-release/2017/6/12/8:59:53.5/" + "log.txt", convert(month, day, hour), (month, day, str(hourC[0]), str(hourC[1]), str(hourC[2])))]

files = []

for fileN in logFiles[-len(logFiles) + 1:]:
	for line in open(fileN[0]).read().split("\n"):
		if "Result:" in line:
			files.append(fileN)
			break

for fileN in files[-amount:]:
	points = {}
	found = False
	fakeData = False
	isTest = False
	counter = 0
	currentTestSet = ""
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
	sets = ["apple","banana","coffeemug","stapler","flashlight","keyboard"]
	#print(points)
	if "testSet" in points and "testSetExclude" in points and len(points["testSetExclude"]) > 1:
		figure = plt.figure()
		fig = figure.add_subplot(1, 1, 1)
		ele = fileN[2]
		plt.title(str(ele[1] + "/" + ele[0] + " " + ele[2] + ":" + ele[3] + ":" + ele[4]))
		x = [e for e in range(0, len(points["testSet"]))]	
		x2 = [e for e in range(0, len(points["testSetExclude"]))]		
		fig.plot(x, points["testSet"])
		fig.plot(x2, points["testSetExclude"])
		for key, point in points.iteritems():
			name = key
			name = name.replace("TestSet", "")
			usedIndex = 0
			if name in sets:
				usedIndex = 20 + sets.index(name) * 10
			else:
				#print(name)
				continue
			x3 = [e for e in range(usedIndex, usedIndex + len(point))]
			fig.plot(x3, point)
	if found:
		print("------------")	

plt.show()
	