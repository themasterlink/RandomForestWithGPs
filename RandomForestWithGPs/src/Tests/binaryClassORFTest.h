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
	std::cout << RED;
	std::cout << "Test size: " << test.size() << std::endl;
	std::cout << "Result:    " << amountOfCorrect / (double) test.size() * 100. << " %" << std::endl;
	std::cout << RESET;
}

void executeForBinaryClassORF(){
	ClassData data;
	ClassData testData;
	DataSets datas;
	TotalStorage::readData(200);
	std::cout << "Finish reading " << std::endl;
	OnlineStorage<ClassPoint*> train;
	OnlineStorage<ClassPoint*> test;
	std::cout << "TotalStorage::getSmallestClassSize(): " << TotalStorage::getSmallestClassSize() << std::endl;
	const int trainAmount = 0.75 * TotalStorage::getSmallestClassSize() * TotalStorage::getAmountOfClass();
	std::cout << "Train amount: " << trainAmount << std::endl;
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

	std::cout << "Amount of Classes: " << TotalStorage::getAmountOfClass() << std::endl;
	if(CommandSettings::get_useFakeData() && (CommandSettings::get_visuRes() > 0 || CommandSettings::get_visuResSimple() > 0)){
		DataWriterForVisu::writeImg("orf.png", &orf, train.storage());
		openFileInViewer("orf.png");
	}
}

#endif /* TESTS_BINARYCLASSORFTEST_H_ */
