/*
 * binaryClassORFTest.h
 *
 *  Created on: 19.10.2016
 *      Author: Max
 */

#ifndef TESTS_MULTICLASSORFIVM_H_
#define TESTS_MULTICLASSORFIVM_H_

#include "../RandomForestGaussianProcess/OnlineRandomForestIVMs.h"
#include "../Utility/Util.h"
#include "../Data/TotalStorage.h"
#include "../Base/CommandSettings.h"

void performTest(OnlineRandomForestIVMs& orf, OnlineStorage<ClassPoint*>& test){
	if(test.size() > 0){
		int amountOfCorrect = 0;
		Labels labels;
		printOnScreen("Predict Data");
		std::cout << std::flush;
		orf.predictData(test.storage(), labels);
		printOnScreen("Predicted Data");
		Eigen::MatrixXd conv = Eigen::MatrixXd::Zero(orf.amountOfClasses(), orf.amountOfClasses());
		for(unsigned int i = 0; i < labels.size(); ++i){
			if(test[i]->getLabel() == labels[i]){
				++amountOfCorrect;
			}
			if(labels[i] != UNDEF_CLASS_LABEL){
				conv.coeffRef(test[i]->getLabel(), labels[i]) += 1;
			}else{
				printError("The label could not be determined!");
			}
		}
		printOnScreen("Test size: " << test.size());
		printOnScreen("Result:    " << amountOfCorrect / (double) test.size() * 100. << " %%");
		ConfusionMatrixPrinter::print(conv);
	}
}

void executeForBinaryClassORFIVM(){

	const int trainAmount = readAllData();
	OnlineStorage<ClassPoint*> train;
	OnlineStorage<ClassPoint*> test;
	int height;
	Settings::getValue("Forest.Trees.height", height);
	OnlineRandomForestIVMs orf(train, height, TotalStorage::getAmountOfClass());
	// starts the training by its own
	TotalStorage::getOnlineStorageCopyWithTest(train, test, trainAmount);
	printOnScreen("Training finished");
	performTest(orf, train);
	printOnScreen("First test finished");

	performTest(orf, test);
	printOnScreen("Second test finished");

	printOnScreen("Amount of Classes: " << TotalStorage::getAmountOfClass());
	if(CommandSettings::get_useFakeData() && (CommandSettings::get_visuRes() > 0 || CommandSettings::get_visuResSimple() > 0)){
		printOnScreen("Print png of data");
		DataWriterForVisu::writeImg("orf.png", &orf, train.storage());
		openFileInViewer("orf.png");
	}
}

#endif /* TESTS_MULTICLASSORFIVM_H_ */
