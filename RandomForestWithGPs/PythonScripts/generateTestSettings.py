#!/usr/bin/python

apple = "0,...,4"
banana = "12,...,15"
coffeemug = "53,...,60"
stapler = "266,...,273"
flashlight = "72,...,76"
keyboard = "152,...,156"
sets = [("apple", apple),("banana",banana),("coffeemug",coffeemug),
        ("stapler",stapler),("flashlight",flashlight), ("keyboard",keyboard)]

def generateTestSettings(amountOfSplits, startCondition, timeFrameUpdate):
    text = "# in python generated test settings file \n"
    text += "load all \n" # load all the information
    exclude = ",".join([apple,banana,coffeemug,stapler,flashlight,keyboard])
    text += "define trainSet without classes {" + exclude + "} from TRAIN_SETTING \n"  # redefine for shorted name
    text += "define testSetExclude without classes {" + exclude + "} from TEST_SETTING \n"  # redefine for shorted name
    text += "define testSet from TEST_SETTING \n"  # redefine for shorted name
    for set1 in sets:
        text += "define " + set1[0] + "Set as classes {" + set1[1] + "} from TRAIN_SETTING \n"
        text += "define " + set1[0] + "SplitSet as 10 splits from " + set1[0] + "Set \n"

    text += "define splitTrainSet as " + str(amountOfSplits) + " splits from trainSet \n\n\n"

    text += "train splitTrainSet[" + str(0) + "] " + startCondition + " with only 6 gb\n"
    text += "test testSetExclude\n"
    text += "test testSet \n"

    for i in range(1, amountOfSplits):
        setNr = i % 20
        if i < 20:
            text += "define actualSet" + str(i) + " from splitTrainSet[" + str(i) + "]\n"
        elif 20 <= i < 30:  # test on the missing classes
            text += "combine appleSplitSet[" + str(setNr) + "] with splitTrainSet[" + str(i) + "] in actualSet" + str(i) + "\n"
        elif 30 <= i < 40:  # test on the missing classes
            text += "combine bananaSplitSet[" + str(setNr) + "] with splitTrainSet[" + str(i) + "] in actualSet" + str(i) + "\n"
        elif 40 <= i < 50:  # test on the missing classes
            text += "combine coffeemugSplitSet[" + str(setNr) + "] with splitTrainSet[" + str(i) + "] in actualSet" + str(i) + "\n"
        elif 50 <= i < 60:  # test on the missing classes
            text += "combine staplerSplitSet[" + str(setNr) + "] with splitTrainSet[" + str(i) + "] in actualSet" + str(i) + "\n"
        elif 60 <= i < 70:  # test on the missing classes
            text += "combine flashlightSplitSet[" + str(setNr) + "] with splitTrainSet[" + str(i) + "] in actualSet" + str(i) + "\n"
        elif 70 <= i:  # test on the missing classes
            text += "combine keyboardSplitSet[" + str(setNr) + "] with splitTrainSet[" + str(i) + "] in actualSet" + str(i) + "\n"

        text += "train actualSet" + str(i) + " for " + str(timeFrameUpdate) + " s with only 6 gb\n"
        text += "test testSetExclude\n"
        text += "test testSet\n"
        if 20 <= i < 30:  # test on the missing classes
            text += "test appleSet \n"
        elif 30 <= i < 40:  # test on the missing classes
            text += "test bananaSet \n"
        elif 40 <= i < 50:  # test on the missing classes
            text += "test coffeemugSet \n"
        elif 50 <= i < 60:  # test on the missing classes
            text += "test staplerSet \n"
        elif 60 <= i < 70:  # test on the missing classes
            text += "test flashlightSet \n"
        elif 70 <= i:  # test on the missing classes
            text += "test keyboardSet \n"

    f = open("../Settings/testSettingsPy.init", "w")
    f.write(text)
    f.close()


amountOfSplits = 80
generateTestSettings(amountOfSplits, "for 1 m", 40)
