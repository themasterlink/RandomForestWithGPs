#!/usr/bin/python


text = "# in python generated test settings file \n"
text += "load all \n" # load all the information
text += "define trainSet from TRAIN_SETTING \n" # redefine for shorted name
text += "define testSet from TEST_SETTING \n" # redefine for shorted name
amountOfSplits = 30
text += "define splitTrainSet as " + str(amountOfSplits) + " splits from trainSet \n"
text += "train splitTrainSet[" + str(0) + "] until 1000 trees with only 3 gb\n"
text += "test testSet \n"
for i in range(1,amountOfSplits):
    text += "train splitTrainSet[" + str(i) + "] with only 3 gb\n"
    text += "test testSet \n"

f = open("../Settings/testSettingsPy.init", "w")
f.write(text);
f.close()