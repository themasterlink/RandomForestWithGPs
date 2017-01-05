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
	Eigen::MatrixXd conv = Eigen::MatrixXd::Zero(orf.amountOfClasses(), orf.amountOfClasses());
	for(unsigned int i = 0; i < labels.size(); ++i){
		if(test[i]->getLabel() == labels[i]){
			++amountOfCorrect;
		}
		conv(test[i]->getLabel(), labels[i]) += 1;
	}
	printOnScreen("Test size: " << test.size());
	printOnScreen("Result:    " << amountOfCorrect / (double) test.size() * 100. << " %");
	ConfusionMatrixPrinter::print(conv);
}

void executeForBinaryClassORF(){
	const int trainAmount = readAllData();
	OnlineStorage<ClassPoint*> train;
	OnlineStorage<ClassPoint*> test;
	int height;
	Settings::getValue("Forest.Trees.height", height);
	OnlineRandomForest orf(train, height, TotalStorage::getAmountOfClass());
	// starts the training by its own
	TotalStorage::getOnlineStorageCopyWithTest(train, test, trainAmount);
	printOnScreen("Training finished");
	performTest(orf, train);
	printOnScreen("First test finished");

	performTest(orf, test);
	printOnScreen("Second test finished");
	orf.update();
	performTest(orf, test);
	printOnScreen("Third test after orf.update() finished");

	printOnScreen("Amount of Classes: " << TotalStorage::getAmountOfClass());
	if(CommandSettings::get_useFakeData() && (CommandSettings::get_visuRes() > 0 || CommandSettings::get_visuResSimple() > 0)){
		DataWriterForVisu::writeImg("orf.png", &orf, train.storage());
		openFileInViewer("orf.png");
	}
}

#endif /* TESTS_BINARYCLASSORFTEST_H_ */
