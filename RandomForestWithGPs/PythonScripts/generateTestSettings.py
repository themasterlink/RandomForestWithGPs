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
    #text += "define trainSet without classes {0,...,150} from trainSet2 \n"
    text += "define testSetExclude without classes {" + exclude + "} from TEST_SETTING \n"  # redefine for shorted name
    text += "define testSet from TEST_SETTING \n"  # redefine for shorted name
    for set1 in sets:
        text += "define " + set1[0] + "Set as classes {" + set1[1] + "} from TRAIN_SETTING \n"
        text += "define " + set1[0] + "TestSet as classes {" + set1[1] + "} from TEST_SETTING \n"
        text += "define " + set1[0] + "SplitSet as 10 splits from " + set1[0] + "Set \n"

    text += "define splitTrainSet as " + str(amountOfSplits) + " splits from trainSet \n\n\n"

    text += "train splitTrainSet[" + str(0) + "] " + startCondition + " with only 6 gb\n"
    text += "test testSetExclude\n"
    text += "test testSet \n"
    tempSetNr = 0
    startPhase = 20
    widthOfUpdate = 10
    for i in range(1, amountOfSplits):
        if i >= startPhase:
            nr = int((i - startPhase) / widthOfUpdate)
            tempSetNr += 1
            text += "combine " + sets[nr][0] + "SplitSet[" + str(i % widthOfUpdate) + "] with splitTrainSet[" + str(i) \
                    + "] in tempSet" + str(tempSetNr) + "\n"
            if nr > 0:
                for t in range(0, nr):
                    tempSetNr += 1
                    text += "combine " + sets[t][0] + "SplitSet[" + str(i % widthOfUpdate) + "] with tempSet" + str(tempSetNr - 1) \
                            + " in tempSet" + str(tempSetNr) + "\n"

            text += "define actualSet" + str(i) + " from tempSet" + str(tempSetNr) + "\n"
        else:
            text += "define actualSet" + str(i) + " from splitTrainSet[" + str(i) + "]\n"

        text += "train actualSet" + str(i) + " " + timeFrameUpdate + " with only 6 gb\n"
        text += "test testSetExclude\n"
        text += "test testSet\n"
        if i >= startPhase:
            nr = int((i - startPhase) / widthOfUpdate)
            for k in range(0, nr + 1):
                text += "test " + sets[k][0] + "TestSet\n"

    f = open("../Settings/testSettingsPy.init", "w")
    f.write(text)
    f.close()


amountOfSplits = 80
generateTestSettings(amountOfSplits, "until 10000 trees", "until 1200 trees")
