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
	StopWatch sw;
	std::vector<std::vector<double> > probs;
	orf.predictData(test.storage(), labels, probs);
	printOnScreen("Needed " << sw.elapsedAsTimeFrame());
	Eigen::MatrixXd conv = Eigen::MatrixXd::Zero(orf.amountOfClasses(), orf.amountOfClasses());
	std::vector<std::list<double> > lists(orf.amountOfClasses(), std::list<double>());
	for(unsigned int i = 0; i < labels.size(); ++i){
		if(labels[i] != UNDEF_CLASS_LABEL){
			if(test[i]->getLabel() == labels[i]){
				++amountOfCorrect;
			}
			lists[labels[i]].push_back(probs[i][labels[i]]); // adds only the winning label to the list
			conv(test[i]->getLabel(), labels[i]) += 1;
		}
	}
	printOnScreen("Test size: " << test.size());
	printOnScreen("Result:    " << amountOfCorrect / (double) test.size() * 100. << " %");
	ConfusionMatrixPrinter::print(conv);

	for(unsigned int i = 0; i < orf.amountOfClasses(); ++i){
		double avg = 0;
		for(std::list<double>::const_iterator it = lists[i].begin(); it != lists[i].end(); ++it){
			avg += *it;
		}
		avg /= lists[i].size();
		printOnScreen("Avg for " << i << " is " << avg << " has " << lists[i].size() << " elements");
		if(CommandSettings::get_visuRes() > 0 || CommandSettings::get_visuResSimple() > 0){
			const std::string fileName = "histForOrf" + number2String(i) + ".svg";
			DataWriterForVisu::writeHisto(fileName, lists[i], 10, 0, 1);
			openFileInViewer(fileName);
		}
	}
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
		StopWatch sw;
		DataWriterForVisu::writeImg("orf.png", &orf, train.storage());
		printOnScreen("For drawing needed time: " << sw.elapsedAsTimeFrame());
		openFileInViewer("orf.png");
	}
}

#endif /* TESTS_BINARYCLASSORFTEST_H_ */
