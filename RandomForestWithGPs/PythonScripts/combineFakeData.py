import os


directory = "../fakeData/"
text = ""
for folder in os.listdir(directory):
	classNr = str(folder)
	filename = directory + folder + "/vectors.txt"
	if os.path.isfile(filename):
		with open(filename, "r") as ins:
			for line in ins:
				line = line.replace("\n", "")
				linesplit = line.split(' ')
				text += ",".join(linesplit) + "," + classNr + "\n"
		ins.close()
with open("../fakeData/allTogether.txt", "w") as ins:
	ins.write(text)