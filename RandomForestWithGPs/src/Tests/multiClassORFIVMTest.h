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

void performTest(OnlineRandomForestIVMs& orf, OnlineStorage<LabeledVectorX*>& test){
	if(test.size() > 0){
		int amountOfCorrect = 0;
		Labels labels;
		printOnScreen("Predict Data");
		std::cout << std::flush;
		std::vector<std::vector<Real> > probs;
		orf.predictData(test.storage(), labels, probs);
		printOnScreen("Predicted Data");
		Matrix conv = Matrix::Zero(orf.amountOfClasses(), orf.amountOfClasses());
		AvgNumber pos, neg;
		for(unsigned int i = 0; i < labels.size(); ++i){
			if(test[i]->getLabel() == labels[i]){
				++amountOfCorrect;
				pos.addNew(probs[i][labels[i]]);
			}else{
				neg.addNew(1.-probs[i][labels[i]]);
			}
			if(labels[i] != UNDEF_CLASS_LABEL){
				conv.coeffRef(test[i]->getLabel(), labels[i]) += 1;
			}else{
				printError("The label could not be determined!");
			}
		}
		printOnScreen("Test size: " << test.size());
		printOnScreen("Result:    " << amountOfCorrect / (Real) test.size() * 100. << " %%");
		printOnScreen("Overconfidence for correct: " << pos.mean() << ", for wrong: " << neg.mean());
		ConfusionMatrixPrinter::print(conv);
	}
}

void executeForBinaryClassORFIVM(){
	const int trainAmount = readAllData();
	if(TotalStorage::instance().getDataSetMode() == TotalStorage::DataSetMode::SEPARATE){
		//	OnlineStorage<LabeledVectorX*> train;
		OnlineStorage<LabeledVectorX*> test;
		int height;
		Settings::instance().getValue("OnlineRandomForest.Tree.height", height);
		const unsigned int amountOfSplits = 5;
		std::vector<OnlineStorage<LabeledVectorX*> > trains(amountOfSplits);
		OnlineRandomForestIVMs orf(trains[0], height, TotalStorage::instance().getAmountOfClass());

		TotalStorage::instance().getOnlineStorageCopySplitsWithTest(trains, test);
		//	TotalStorage::instance().getOnlineStorageCopyWithTest(train, test, trainAmount);
		printOnScreen("Training finished");
		performTest(orf, trains[0]);
		printOnScreen("First test finished");
		for(unsigned int i = 1; i < amountOfSplits; ++i){
			// on test set!
			printOnScreen("On test set:");
			performTest(orf, test);
			printOnScreen("Done on test set");

			printOnScreen("On next training set:");
			performTest(orf, trains[i]);
			printOnScreen("Done on next training set");
			// filter out the wrong ones
			LabeledData wrongOnes;
			wrongOnes.reserve(trains[i].size());
			Labels labels;
			orf.predictData(trains[i].storage(), labels);
			std::vector<unsigned int> classCounter(orf.amountOfClasses(), 0);
			for(unsigned int j = 0; j < trains[i].size(); ++j){
				if(labels[j] != trains[i][j]->getLabel()){
					wrongOnes.push_back(trains[i][j]);
					classCounter[labels[j]] += 1;
				}
			}
			for(unsigned int j = 0; j < orf.amountOfClasses(); ++j){
				printOnScreen("Class: " << j << " with: " << classCounter[j]);
			}
			trains[0].append(wrongOnes);
			printOnScreen("Trained: " << i << " dataset");
		}

		int testNr = 2;
		Settings::instance().getValue("TotalStorage.folderTestNr", testNr);

		performTest(orf, test);
		printOnScreen("Second test finished");
		//		orf.update();
		//		performTest(orf, test);
		//		printOnScreen("Third test after orf.update() finished");

		printOnScreen("Amount of Classes: " << TotalStorage::instance().getAmountOfClass());
	}else{
		OnlineStorage<LabeledVectorX*> train;
		OnlineStorage<LabeledVectorX*> test;
		int height;
		Settings::instance().getValue("OnlineRandomForest.Tree.height", height);
		OnlineRandomForestIVMs orf(train, height, TotalStorage::instance().getAmountOfClass());
		// starts the training by its own

//		std::vector<unsigned int> counter(10,0);
		TotalStorage::instance().getOnlineStorageCopyWithTest(train, test, trainAmount);
//		for(OnlineStorage<LabeledVectorX*>::Iterator it = train.begin(); it != train.end(); ++it){
//			counter[(*it)->getLabel()] += 1;
//		}
//		for(unsigned int i = 0; i < 10; ++i){
//			printOnScreen("i: " << i << ", " << counter[i]);
//		}
//		sleepFor(5);
//
		printOnScreen("Training finished");
		performTest(orf, train);
		printOnScreen("First test finished");

		performTest(orf, test);
		printOnScreen("Second test finished");

		printOnScreen("Amount of Classes: " << TotalStorage::instance().getAmountOfClass());
		if(CommandSettings::instance().get_useFakeData() &&
		   (CommandSettings::instance().get_visuRes() > 0 || CommandSettings::instance().get_visuResSimple() > 0)){
			printOnScreen("Print png of data");
			DataWriterForVisu::writeImg("orf.png", &orf, train.storage());
			openFileInViewer("orf.png");
		}
	}
}

#endif /* TESTS_MULTICLASSORFIVM_H_ */
