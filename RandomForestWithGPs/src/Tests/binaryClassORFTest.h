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

void executeForBinaryClassORF(const std::string& path, const bool useRealData, const bool visu){
	ClassData data;
	ClassData testData;
	DataSets datas;
	TotalStorage::readData(200, useRealData);
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

	for(unsigned int k = 0; k < 100; ++k){
		bool done = orf.update();
		performTest(orf, test);
		if(!done){break;};
	}
	std::cout << "Amount of Classes: " << TotalStorage::getAmountOfClass() << std::endl;
	if(!useRealData && visu){
		DataWriterForVisu::writeSvg("orf.svg", &orf, 150, train.storage());
		openFileInViewer("orf.svg");
	}
}

#endif /* TESTS_BINARYCLASSORFTEST_H_ */
