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

void performTest(OnlineRandomForest& orf, OnlineStorage<LabeledVectorX*>& test){
	if(test.size() == 0){
		return;
	}
	int amountOfCorrect = 0;
	Labels labels;
	StopWatch sw;
	std::vector<std::vector<Real> > probs;
	orf.predictData(test.storage(), labels, probs);
	printOnScreen("Needed " << sw.elapsedAsTimeFrame());
	Matrix conv = Matrix::Zero(orf.amountOfClasses(), orf.amountOfClasses());
//	std::vector<std::list<Real> > lists(orf.amountOfClasses(), std::list<Real>());
	AvgNumber oc, uc;
	AvgNumber ocBVS, ucBVS;
	const unsigned int amountOfClasses = ClassKnowledge::instance().amountOfClasses();
	const Real logBase = logReal(amountOfClasses);
	for(unsigned int i = 0; i < labels.size(); ++i){
		if(labels[i] != UNDEF_CLASS_LABEL){
			Real entropy = 0;
			for(unsigned int j = 0; j < amountOfClasses; ++j){
				if(probs[i][j] > 0){
					entropy -= probs[i][j] * logReal(probs[i][j]) / logBase;
				}
			}
			Real max1 = 0, max2 = 0;
			for(unsigned int j = 0; j < amountOfClasses; ++j){
				if(probs[i][j] > max1){
					max2 = max1;
					max1 = probs[i][j];
				}
			}
			Real entropyBVS = max2 / max1;
			if(test[i]->getLabel() == labels[i]){
				++amountOfCorrect;
				uc.addNew(entropy);
				ucBVS.addNew(entropyBVS);
			}else{
				oc.addNew(1.-entropy);
				ocBVS.addNew(1.-entropyBVS);
//				printOnScreen("Class: " << ClassKnowledge::instance().getNameFor(test[i]->getLabel()) << ", for 0: " << probs[i][0] << ", for 1: " << probs[i][1]);
			}
//			lists[labels[i]].push_back(probs[i][labels[i]]); // adds only the winning label to the list
			conv(test[i]->getLabel(), labels[i]) += 1;

		}
	}
	printOnScreen("Test size: " << test.size());
	printOnScreen("Result:    " << amountOfCorrect / (Real) test.size() * 100. << " %");
	printOnScreen("Overconf:  " << oc.mean() * 100.0 << "%%");
	printOnScreen("Underconf: " << uc.mean() * 100.0 << "%%");
	printOnScreen("Overconf BVS:  " << ocBVS.mean() * 100.0 << "%%");
	printOnScreen("Underconf BVS: " << ucBVS.mean() * 100.0 << "%%");
	if(conv.rows() < 40){ // otherwise not useful
		ConfusionMatrixPrinter::print(conv);
	}
//	for(unsigned int i = 0; i < orf.amountOfClasses(); ++i){
//		Real avg = 0;
//		for(std::list<Real>::const_iterator it = lists[i].begin(); it != lists[i].end(); ++it){
//			avg += *it;
//		}
//		avg /= lists[i].size();
//		printOnScreen("Avg for " << i << " is " << avg << " has " << lists[i].size() << " elements");
//		if(CommandSettings::instance().get_visuRes() > 0 || CommandSettings::instance().get_visuResSimple() > 0){
//			const std::string fileName = "histForOrf" + StringHelper::number2String(i) + ".svg";
//			DataWriterForVisu::writeHisto(fileName, lists[i], 10, 0, 1);
//			openFileInViewer(fileName);
//		}
//	}
}

void executeForBinaryClassORF(){
	const int trainAmount = readAllData();
	if(TotalStorage::instance().getDataSetMode() == TotalStorage::DataSetMode::SEPARATE){
		UniquePtr<OnlineRandomForest> newOrf;
		if(false){
			//	OnlineStorage<LabeledVectorX*> train;
			OnlineStorage<LabeledVectorX*> test;
			int height;
			Settings::instance().getValue("OnlineRandomForest.Tree.height", height);
			const unsigned int amountOfSplits = 10;
			std::vector<OnlineStorage<LabeledVectorX*> > trains(amountOfSplits);
			newOrf = std::make_unique<OnlineRandomForest>(trains[0], (unsigned int) height,
														  TotalStorage::instance().getAmountOfClass());
			// starts the training by its own
			//	TotalStorage::instance().getOnlineStorageCopySplitsWithTest(trains, test);
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
			//	sleepFor(5);
			//	quitApplication();


			TotalStorage::instance().getOnlineStorageCopySplitsWithTest(trains, test);
			printOnScreen("point: " << trains[0][0]->transpose());
			//	TotalStorage::instance().getOnlineStorageCopyWithTest(train, test, trainAmount);
			printOnScreen("Training finished");
			auto& orf = *newOrf.get();
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

			printOnScreen("Amount of Classes: " << TotalStorage::instance().getAmountOfClass());
		}else{
			OnlineStorage<LabeledVectorX*> test;
			int height;
			Settings::instance().getValue("OnlineRandomForest.Tree.height", height);
			OnlineStorage<LabeledVectorX*> train;
			newOrf = std::make_unique<OnlineRandomForest>(train, (unsigned int) height,
														  TotalStorage::instance().getAmountOfClass());
			auto& orf = *newOrf;

			TotalStorage::instance().getOnlineStorageCopyWithTest(train, test, 10000000);
			printOnScreen("Training finished");
			performTest(orf, train);
//
//			printOnScreen("Test on training set finished");
			performTest(orf, test);
			printOnScreen("Test on test set finished");
		}
		unsigned int removedClass;
		Settings::instance().getValue("TotalStorage.excludeClass", removedClass);
		if(ClassKnowledge::instance().hasClassName(removedClass) && newOrf != nullptr){
			printOnScreen("Removed class: " << removedClass);
			OnlineStorage<LabeledVectorX*> removedTrain;
			OnlineStorage<LabeledVectorX*> removedTest;
			TotalStorage::instance().getRemovedOnlineStorageCopyWithTest(removedTrain, removedTest);
			printOnScreen("On " << removedTrain.size() << " removed points from trainings data:");
			performTest(*newOrf, removedTrain);
			printOnScreen("On " << removedTest.size() << " removed points from real test data:");
			performTest(*newOrf, removedTest);
		}
	}else{
		OnlineStorage<LabeledVectorX*> train;
		OnlineStorage<LabeledVectorX*> test;
		unsigned int height;
		Settings::instance().getValue("OnlineRandomForest.Tree.height", height);
		OnlineRandomForest orf(train, height, TotalStorage::instance().getAmountOfClass());
		TotalStorage::instance().getOnlineStorageCopyWithTest(train, test, trainAmount);
		printOnScreen("Training finished");
		performTest(orf, train);
		printOnScreen("First test finished");
		performTest(orf, test);
//		printOnScreen("Second test finished");
//		orf.update();
//		performTest(orf, test);
		printOnScreen("Third test after orf.update() finished");
		printOnScreen("Amount of Classes: " << TotalStorage::instance().getAmountOfClass());
//		DataPoint p;
//		p.resize(2);
//		p[0] = 3.0;
//		p[1] = 0.2;
//		const auto i = orf.predict(p);

		if(CommandSettings::instance().get_useFakeData() &&
		   (CommandSettings::instance().get_visuRes() > 0 || CommandSettings::instance().get_visuResSimple() > 0)){
			StopWatch sw;
			DataWriterForVisu::writeImg("orf.png", &orf, train.storage());
			printOnScreen("For drawing needed time: " << sw.elapsedAsTimeFrame());
			openFileInViewer("orf.png");
		}
	}
	printOnScreen("Reached end of binary ORF");
	sleepFor(2);
}

#endif /* TESTS_BINARYCLASSORFTEST_H_ */
