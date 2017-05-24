
import os

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
		newPath = path + month + "/" + day
		for hour in os.listdir(newPath):
			if os.path.isfile(newPath + "/" + hour + "/" + "log.txt"):
				hourC = hourConvert(hour)
				t = (newPath + "/" + hour + "/" + "log.txt", convert(month, day, hour), (month, day, str(hourC[0]), str(hourC[1]), str(hourC[2])))
				logFiles.append(t)

logFiles.sort(key=lambda tup: tup[1])
amount = int(raw_input("Nr of last log files: "))
for fileN in logFiles[-amount:]:
	found = False
	fakeData = False
	isTest = False
	for line in open(fileN[0]).read().split("\n"):
		if "fakeData" in line:
			fakeData = True
		if "Done on test set" in line:
			isTest = False
		if "On test set" in line:
			isTest = True
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
				print(str(ele[1] + "/" + ele[0] + " " + ele[2] + ":" + ele[3] + ":" + ele[4] + " " + color + line + "\033[0m"))
	if found:
		print("------------")	

