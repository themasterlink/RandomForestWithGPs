/*
 * binaryClassIVMTest.h
 *
 *  Created on: 04.10.2016
 *      Author: Max
 */

#ifndef TESTS_MULTICLASSIVMTEST_H_
#define TESTS_MULTICLASSIVMTEST_H_

#include <Eigen/Dense>
#include "../Data/TotalStorage.h"
#include "../Data/DataConverter.h"
#include "../Data/DataWriterForVisu.h"
#include "../GaussianProcess/IVMMultiBinary.h"
#include "../GaussianProcess/BayesOptimizerIVM.h"
#include "../Base/Settings.h"
#include "../Utility/ConfusionMatrixPrinter.h"
#include "../Base/CommandSettings.h"
#include <chrono>
#include <thread>

void testIvm(IVMMultiBinary& ivms, const ClassData& data){
	int right = 0;
	const int amountOfTestPoints = data.size();
//	Eigen::Vector2i rightPerClass;
//	rightPerClass[0] = rightPerClass[1] = 0;
//	int amountOfBelow = 0;
//	int amountOfAbove = 0;
//	std::list<double> probs;
//	Eigen::Vector2i amountPerClass;
//	amountPerClass[0] = amountPerClass[1] = 0;
	Eigen::MatrixXd conv = Eigen::MatrixXd::Zero(ivms.amountOfClasses(), ivms.amountOfClasses());
	for(int i = 0; i < amountOfTestPoints; ++i){
		const int label = ivms.predict(*data[i]);
		if(label == data[i]->getLabel()){
			++right;
		}
		conv(data[i]->getLabel(), label) += 1;
		/*if(data[i]->getLabel() == label){
			++amountPerClass[label];
		}else if(data[i]->getLabel() == ivms.getLabelForMinusOne()){
			++amountPerClass[1];
		}
		if(prob > 0.5 && data[i]->getLabel() == ivms.getLabelForOne()){
			++right; ++rightPerClass[0];
		}else if(prob < 0.5 && data[i]->getLabel() == ivms.getLabelForMinusOne()){
			++right; ++rightPerClass[1];
		}
		if(prob > 0.5){
			++amountOfAbove;
		}else if(prob < 0.5){
			++amountOfBelow;
		}
		probs.push_back(prob);*/
	}
//	if(amountOfTestPoints > 0 && CommandSettings::get_plotHistos()){
//		DataWriterForVisu::writeHisto("histo.svg", probs, 14, 0, 1);
//		openFileInViewer("histo.svg");
//	}
	ConfusionMatrixPrinter::print(conv, std::cout);
	std::cout << RED;
	std::cout << "Amount of right: " << (double) right / amountOfTestPoints * 100.0 << "%" << std::endl;
//	std::cout << "Amount of above: " << (double) amountOfAbove / amountOfTestPoints * 100.0 << "%" << std::endl;
//	std::cout << "Amount of below: " << (double) amountOfBelow / amountOfTestPoints * 100.0 << "%" << std::endl;
//	std::cout << "Recall for  1: " << (double) rightPerClass[0] / (double) amountPerClass[0] * 100.0 << "%" << std::endl;
//	std::cout << "Recall for -1: " << (double) rightPerClass[1] / (double) amountPerClass[1] * 100.0 << "%" << std::endl;
//	std::cout << "Precision for  1: " << (double) rightPerClass[0] / right * 100.0 << "%" << std::endl;
//	std::cout << "Precision for -1: " << (double) rightPerClass[1] / right * 100.0 << "%" << std::endl;
//	std::cout << "Amount of 1 in total: " << (double) amountPerClass[0] / amountOfTestPoints * 100.0 << "%" << std::endl;
//	std::cout << ivm.getKernel().prettyString() << std::endl;
	std::cout << RESET;
}

void executeForMutliClassIVM(){
	const int firstPoints = 10000000; // all points
	TotalStorage::readData(firstPoints);
	DataSets datas;
	std::cout << "TotalStorage::getSmallestClassSize(): " << TotalStorage::getSmallestClassSize() << std::endl;
	const int trainAmount = 0.75 * TotalStorage::getSmallestClassSize() * TotalStorage::getAmountOfClass();
	OnlineStorage<ClassPoint*> train;
	OnlineStorage<ClassPoint*> test;
	std::cout << "Finish reading " << std::endl;

	bool doEpUpdate;
	Settings::getValue("IVM.doEpUpdate", doEpUpdate);
	int number;
	Settings::getValue("IVM.nrOfInducingPoints", number);
	IVMMultiBinary ivms(train, number, doEpUpdate);

	// starts the training by its own
	TotalStorage::getOnlineStorageCopyWithTest(train, test, trainAmount);
	std::cout << "Finish training" << std::endl;

	std::cout << "On " << train.storage().size() << " points from trainings data:" << std::endl;
	testIvm(ivms, train.storage());
	std::cout << "On " << test.storage().size() << " points from real test data:" << std::endl;
	testIvm(ivms, test.storage());

	if(CommandSettings::get_useFakeData() && (CommandSettings::get_visuRes() > 0 || CommandSettings::get_visuResSimple() > 0)){
		DataWriterForVisu::writeSvg("ivms.svg", &ivms, train.storage());
		system("open ivms.svg");
		DataWriterForVisu::writeImg("ivms.png", &ivms, train.storage());
		system("open ivms.png");
	}
}



#endif /* TESTS_MULTICLASSIVMTEST_H_ */
