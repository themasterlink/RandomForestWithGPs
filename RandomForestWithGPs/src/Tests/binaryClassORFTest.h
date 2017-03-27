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
	if(test.size() == 0){
		return;
	}
	int amountOfCorrect = 0;
	Labels labels;
	StopWatch sw;
	std::vector<std::vector<double> > probs;
	orf.predictData(test.storage(), labels, probs);
	printOnScreen("Needed " << sw.elapsedAsTimeFrame());
	Eigen::MatrixXd conv = Eigen::MatrixXd::Zero(orf.amountOfClasses(), orf.amountOfClasses());
//	std::vector<std::list<double> > lists(orf.amountOfClasses(), std::list<double>());
	AvgNumber oc, uc;
	AvgNumber ocBVS, ucBVS;
	const unsigned int amountOfClasses = ClassKnowledge::amountOfClasses();
	const double logBase = log(amountOfClasses);
	for(unsigned int i = 0; i < labels.size(); ++i){
		if(labels[i] != UNDEF_CLASS_LABEL){
			double entropy = 0;
			for(unsigned int j = 0; j < amountOfClasses; ++j){
				if(probs[i][j] > 0){
					entropy -= probs[i][j] * log(probs[i][j]) / logBase;
				}
			}
			double max1 = 0, max2 = 0;
			for(unsigned int j = 0; j < amountOfClasses; ++j){
				if(probs[i][j] > max1){
					max2 = max1;
					max1 = probs[i][j];
				}
			}
			double entropyBVS = max2 / max1;
			if(test[i]->getLabel() == labels[i]){
				++amountOfCorrect;
				uc.addNew(entropy);
				ucBVS.addNew(entropyBVS);
			}else{
				oc.addNew(1.-entropy);
				ocBVS.addNew(1.-entropyBVS);
//				printOnScreen("Class: " << ClassKnowledge::getNameFor(test[i]->getLabel()) << ", for 0: " << probs[i][0] << ", for 1: " << probs[i][1]);
			}
//			lists[labels[i]].push_back(probs[i][labels[i]]); // adds only the winning label to the list
			conv(test[i]->getLabel(), labels[i]) += 1;

		}
	}
	printOnScreen("Test size: " << test.size());
	printOnScreen("Result:    " << amountOfCorrect / (double) test.size() * 100. << " %");
	printOnScreen("Overconf:  " << (double) oc.mean() * 100.0 << "%%");
	printOnScreen("Underconf: " << (double) uc.mean() * 100.0 << "%%");
	printOnScreen("Overconf BVS:  " << (double) ocBVS.mean() * 100.0 << "%%");
	printOnScreen("Underconf BVS: " << (double) ucBVS.mean() * 100.0 << "%%");

	ConfusionMatrixPrinter::print(conv);

//	for(unsigned int i = 0; i < orf.amountOfClasses(); ++i){
//		double avg = 0;
//		for(std::list<double>::const_iterator it = lists[i].begin(); it != lists[i].end(); ++it){
//			avg += *it;
//		}
//		avg /= lists[i].size();
//		printOnScreen("Avg for " << i << " is " << avg << " has " << lists[i].size() << " elements");
//		if(CommandSettings::get_visuRes() > 0 || CommandSettings::get_visuResSimple() > 0){
//			const std::string fileName = "histForOrf" + number2String(i) + ".svg";
//			DataWriterForVisu::writeHisto(fileName, lists[i], 10, 0, 1);
//			openFileInViewer(fileName);
//		}
//	}
}

void executeForBinaryClassORF(){
	const int trainAmount = readAllData();
	if(TotalStorage::getMode() == TotalStorage::SEPERATE){
		OnlineRandomForest* newOrf = nullptr;
		if(false){
			//	OnlineStorage<ClassPoint*> train;
			OnlineStorage<ClassPoint*> test;
			int height;
			Settings::getValue("Forest.Trees.height", height);
			const unsigned int amountOfSplits = 10;
			std::vector<OnlineStorage<ClassPoint*> > trains(amountOfSplits);
			newOrf = new OnlineRandomForest(trains[0], height, TotalStorage::getAmountOfClass());
			OnlineRandomForest& orf = *newOrf;
			// starts the training by its own
			//	TotalStorage::getOnlineStorageCopySplitsWithTest(trains, test);
			//	const unsigned int dim =  trains[0].dim();
			//	for(unsigned int m = 0; m < amountOfSplits; ++m){
			//		printOnScreen("m: " << m);
			//		for(unsigned int i = 0; i < trains[m].size(); ++i){
			//			for(unsigned int l = 0; l < amountOfSplits; ++l){
			//				if(m != l){
			//					for(unsigned int j = 0; j < trains[l].size(); ++j){
			//						bool diff = true;
			//						for(unsigned int k = 0; k < dim; ++k){
			//							if(fabs(trains[m][i]->coeff(k) - trains[l][j]->coeff(k)) > 1e-7){
			//								diff = false;
			//								break;
			//							}
			//						}
			//						if(diff){
			//							printError("Are the same!");
			//						}
			//					}
			//				}
			//			}
			//		}
			//	}
			//	sleep(5);
			//	exit(0);


			TotalStorage::getOnlineStorageCopySplitsWithTest(trains, test);
			//	TotalStorage::getOnlineStorageCopyWithTest(train, test, trainAmount);
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
				ClassData wrongOnes;
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
//				for(unsigned int j = 0; j < orf.amountOfClasses(); ++j){
//					printOnScreen("Class: " << j << " with: " << classCounter[j]);
//				}
				StopWatch sw2;
				trains[0].append(wrongOnes);

				printOnScreen("Trained: " << i << " dataset, in " << sw2.elapsedAsTimeFrame());
			}

			performTest(orf, test);
			printOnScreen("Second test finished");
			orf.update();
			performTest(orf, test);
			printOnScreen("Third test after orf.update() finished");

			printOnScreen("Amount of Classes: " << TotalStorage::getAmountOfClass());
		}else{
			OnlineStorage<ClassPoint*> test;
			int height;
			Settings::getValue("Forest.Trees.height", height);
			OnlineStorage<ClassPoint*> train;
			newOrf = new OnlineRandomForest(train, height, TotalStorage::getAmountOfClass());
			OnlineRandomForest& orf = *newOrf;

			TotalStorage::getOnlineStorageCopyWithTest(train, test, 10000000);
			printOnScreen("Training finished");
//			performTest(orf, train);
//
//			printOnScreen("Test on training set finished");
			performTest(orf, test);
			printOnScreen("Test on test set finished");
		}
		int removedClass;
		Settings::getValue("TotalStorage.excludeClass",removedClass);
		if(ClassKnowledge::hasClassName(removedClass) && newOrf != nullptr){
			printOnScreen("Removed class: " << removedClass);
			OnlineStorage<ClassPoint*> removedTrain;
			OnlineStorage<ClassPoint*> removedTest;
			TotalStorage::getRemovedOnlineStorageCopyWithTest(removedTrain, removedTest);
			printOnScreen("On " << removedTrain.size() << " removed points from trainings data:");
			performTest(*newOrf, removedTrain);
			printOnScreen("On " << removedTest.size() << " removed points from real test data:");
			performTest(*newOrf, removedTest);
		}
	}else{
		OnlineStorage<ClassPoint*> train;
		OnlineStorage<ClassPoint*> test;
		int height;
		Settings::getValue("Forest.Trees.height", height);
		OnlineRandomForest orf(train, height, TotalStorage::getAmountOfClass());
		TotalStorage::getOnlineStorageCopyWithTest(train, test, trainAmount);
		printOnScreen("Training finished");
		performTest(orf, train);
		printOnScreen("First test finished");
		performTest(orf, test);
//		printOnScreen("Second test finished");
//		orf.update();
//		performTest(orf, test);
		printOnScreen("Third test after orf.update() finished");
		printOnScreen("Amount of Classes: " << TotalStorage::getAmountOfClass());

		if(CommandSettings::get_useFakeData() && (CommandSettings::get_visuRes() > 0 || CommandSettings::get_visuResSimple() > 0)){
			StopWatch sw;
			DataWriterForVisu::writeImg("orf.png", &orf, train.storage());
			printOnScreen("For drawing needed time: " << sw.elapsedAsTimeFrame());
			openFileInViewer("orf.png");
		}
	}

}

#endif /* TESTS_BINARYCLASSORFTEST_H_ */
