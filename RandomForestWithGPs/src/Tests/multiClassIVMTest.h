/*
 * binaryClassIVMTest.h
 *
 *  Created on: 04.10.2016
 *      Author: Max
 */

#ifndef TESTS_MULTICLASSIVMTEST_H_
#define TESTS_MULTICLASSIVMTEST_H_

#include "../Data/TotalStorage.h"
#include "../Data/DataConverter.h"
#include "../Data/DataWriterForVisu.h"
#include "../GaussianProcess/IVMMultiBinary.h"
#include "../GaussianProcess/BayesOptimizerIVM.h"
#include "../Base/Settings.h"
#include "../Base/ScreenOutput.h"
#include "../Utility/ConfusionMatrixPrinter.h"
#include "../Base/CommandSettings.h"
#include <chrono>
#include <thread>

void testIvm(IVMMultiBinary& ivms, const LabeledData& data){
	const int amountOfTestPoints = data.size();
	int right = 0;
//	Vector2i rightPerClass;
//	rightPerClass[0] = rightPerClass[1] = 0;
//	int amountOfBelow = 0;
//	int amountOfAbove = 0;
//	std::list<Real> probs;
//	Vector2i amountPerClass;
//	amountPerClass[0] = amountPerClass[1] = 0;
	Matrix conv = Matrix::Zero(ivms.amountOfClasses(), ivms.amountOfClasses());
	Labels labels;
	std::vector< std::vector<Real> > probs;
	ivms.predictData(data, labels, probs);
	const unsigned int amountOfClasses = ClassKnowledge::instance().amountOfClasses();
	const Real logBase = logReal(amountOfClasses);
	AvgNumber oc, uc;
	AvgNumber ocBVS, ucBVS;
	for(int i = 0; i < amountOfTestPoints; ++i){
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
		const unsigned int label = labels[i];
		if(label == data[i]->getLabel()){
			uc.addNew(entropy);
			ucBVS.addNew(entropyBVS);
			++right;
		}else{
			oc.addNew(1.-entropy);
			ocBVS.addNew(1.-entropyBVS);
		}
		if(label != UNDEF_CLASS_LABEL){
			conv(data[i]->getLabel(), label) += 1;
		}
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
//	if(amountOfTestPoints > 0 && CommandSettings::instance().get_plotHistos()){
//		DataWriterForVisu::writeHisto("histo.svg", probs, 14, 0, 1);
//		openFileInViewer("histo.svg");
//	}
	ConfusionMatrixPrinter::print(conv);
	printOnScreen("Overconf:  " << (Real) oc.mean() * 100.0 << "%%");
	printOnScreen("Underconf: " << (Real) uc.mean() * 100.0 << "%%");
	printOnScreen("Overconf BVS:  " << (Real) ocBVS.mean() * 100.0 << "%%");
	printOnScreen("Underconf BVS: " << (Real) ucBVS.mean() * 100.0 << "%%");
	printOnScreen("Amount of right: " << (Real) right / amountOfTestPoints * 100.0 << "%%");
//	std::cout << "Amount of above: " << (Real) amountOfAbove / amountOfTestPoints * 100.0 << "%" << std::endl;
//	std::cout << "Amount of below: " << (Real) amountOfBelow / amountOfTestPoints * 100.0 << "%" << std::endl;
//	std::cout << "Recall for  1: " << (Real) rightPerClass[0] / (Real) amountPerClass[0] * 100.0 << "%" << std::endl;
//	std::cout << "Recall for -1: " << (Real) rightPerClass[1] / (Real) amountPerClass[1] * 100.0 << "%" << std::endl;
//	std::cout << "Precision for  1: " << (Real) rightPerClass[0] / right * 100.0 << "%" << std::endl;
//	std::cout << "Precision for -1: " << (Real) rightPerClass[1] / right * 100.0 << "%" << std::endl;
//	std::cout << "Amount of 1 in total: " << (Real) amountPerClass[0] / amountOfTestPoints * 100.0 << "%" << std::endl;
//	std::cout << ivm.getKernel().prettyString() << std::endl;
//	std::cout << RESET;
	std::fstream output;
	output.open(Logger::instance().getActDirectory() + "resultForEachPoint.csv",
				std::fstream::out | std::fstream::trunc);
	std::vector<int> classCounter(ivms.amountOfClasses(),0);
	if(output.is_open()){
		output << "real;predicted";
		for(unsigned int i = 0; i < ivms.amountOfClasses(); ++i){
			output << ";" << i;
		}
		output << "\n";
		for(int i = 0; i < amountOfTestPoints; ++i){
			if(classCounter[data[i]->getLabel()] < 100){
				++classCounter[data[i]->getLabel()];
				output << data[i]->getLabel() << ";" << labels[i];
				for(unsigned int j = 0; j < ivms.amountOfClasses(); ++j){
					output << ";" << StringHelper::number2String(probs[i][j], 5);
				}
				output << "\n";
			}
		}
	}
	output.close();
	std::string out = "";
	std::ifstream input(Logger::instance().getActDirectory() + "resultForEachPoint.csv");
	if(input.is_open()){
		std::string line;
		while(std::getline(input, line)){
			for(unsigned int i = 0; i < line.length(); ++i){
				if(line[i] == '.'){
					line[i] = ',';
				}
			}
			out += line + "\n";
		}
	}
	std::fstream output2;
	output2.open(Logger::instance().getActDirectory() + "resultForEachPoint.csv",
				 std::fstream::out | std::fstream::trunc);
	output2.write(out.c_str(), out.length());
	output2.close();
}

void executeForMutliClassIVM(){
	const int trainAmount = readAllData();
	OnlineStorage<LabeledVectorX*> train;
	OnlineStorage<LabeledVectorX*> test;
	printOnScreen("Finish reading");

	bool doEpUpdate;
	Settings::instance().getValue("IVM.doEpUpdate", doEpUpdate);
	int number;
	Settings::instance().getValue("IVM.nrOfInducingPoints", number);
	IVMMultiBinary ivms(train, number, doEpUpdate);
	// starts the training by its own
	TotalStorage::instance().getOnlineStorageCopyWithTest(train, test, trainAmount);
	printOnScreen("Finish training");

	printOnScreen("On " << train.storage().size() << " points from trainings data:");
	testIvm(ivms, train.storage());
	printOnScreen("On " << test.storage().size() << " points from real test data:");
	testIvm(ivms, test.storage());

	if(CommandSettings::instance().get_useFakeData() &&
	   (CommandSettings::instance().get_visuRes() > 0 || CommandSettings::instance().get_visuResSimple() > 0)){
		DataWriterForVisu::writeSvg("ivms.svg", &ivms, train.storage());
		openFileInViewer("ivms.svg");
//		DataWriterForVisu::writeImg("ivms.png", &ivms, train.storage());
//		openFileInViewer("ivms.png");
	}
}



#endif /* TESTS_MULTICLASSIVMTEST_H_ */
