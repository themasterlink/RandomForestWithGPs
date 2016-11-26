/*
 * binaryClassORFTest.h
 *
 *  Created on: 19.10.2016
 *      Author: Max
 */

#ifndef TESTS_BINARYCLASSORFTEST_H_
#define TESTS_BINARYCLASSORFTEST_H_

#include "../RandomForests/OnlineRandomForest.h"
#include "../Utility/Util.h"
#include "../Data/TotalStorage.h"
#include "../Base/CommandSettings.h"

void performTest(OnlineRandomForest& orf, OnlineStorage<ClassPoint*>& test){
	int amountOfCorrect = 0;
	Labels labels;
	orf.predictData(test.storage(), labels);
	for(unsigned int i = 0; i < labels.size(); ++i){
		if(test[i]->getLabel() == labels[i]){
			++amountOfCorrect;
		}
	}
	printOnScreen("Test size: " << test.size());
	printOnScreen("Result:    " << amountOfCorrect / (double) test.size() * 100. << " %");
}

void executeForBinaryClassORF(){
	ClassData data;
	ClassData testData;
	DataSets datas;
	TotalStorage::readData(200);
	printOnScreen("Finish reading ");
	OnlineStorage<ClassPoint*> train;
	OnlineStorage<ClassPoint*> test;
	printOnScreen("TotalStorage::getSmallestClassSize(): " << TotalStorage::getSmallestClassSize());
	const int trainAmount = 0.75 * TotalStorage::getSmallestClassSize() * TotalStorage::getAmountOfClass();
	printOnScreen("Train amount: " << trainAmount);
	int amountOfTrees, height;
	Settings::getValue("Forest.amountOfTrees", amountOfTrees);
	Settings::getValue("Forest.Trees.height", height);
	OnlineRandomForest orf(train, height, amountOfTrees, TotalStorage::getAmountOfClass());
	// starts the training by its own
	TotalStorage::getOnlineStorageCopyWithTest(train, test, trainAmount);
	performTest(orf, train);

	performTest(orf, test);
	orf.update();
	performTest(orf, test);

	printOnScreen("Amount of Classes: " << TotalStorage::getAmountOfClass());
	if(CommandSettings::get_useFakeData() && (CommandSettings::get_visuRes() > 0 || CommandSettings::get_visuResSimple() > 0)){
		DataWriterForVisu::writeImg("orf.png", &orf, train.storage());
		openFileInViewer("orf.png");
	}
}

#endif /* TESTS_BINARYCLASSORFTEST_H_ */
