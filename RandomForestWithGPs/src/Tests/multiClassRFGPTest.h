/*
 * multiClassRFGPTest.h
 *
 *  Created on: 15.09.2016
 *      Author: Max
 */

#ifndef TESTS_MULTICLASSRFGPTEST_H_
#define TESTS_MULTICLASSRFGPTEST_H_

#include <Eigen/Dense>

#include "../Base/Settings.h"
#include "../Data/DataReader.h"
/*#include "../Data/DataConverter.h"
#include "../Data/DataWriterForVisu.h"
#include "../GaussianProcess/GaussianProcess.h"
#include "../GaussianProcess/GaussianProcessMultiBinary.h"
*/
#include "../RandomForestGaussianProcess/RandomForestGaussianProcess.h"
#include "../RandomForestGaussianProcess/RFGPWriter.h"
//#include "../Utility/ConfusionMatrixPrinter.h"

void executeForRFGPMultiClass(const std::string& path){

	//testSpeedOfEigenMultWithDiag();
//	executeForRFBinaryClass();
//	return 0;

	//executeForRFBinaryClass();
	//return 0;

	DataSets dataSets;
	DataReader::readFromFiles(dataSets, path, 500);
	DataSets trainSets;
	DataSets testSets;
	const double facForTraining = 0.8;
	for(DataSetsConstIterator it = dataSets.begin(); it != dataSets.end(); ++it){
		trainSets.insert(DataSetPair(it->first, ClassData()));
		DataSetsIterator itTrain = trainSets.find(it->first);
		itTrain->second.reserve(facForTraining * it->second.size());
		testSets.insert(DataSetPair(it->first, ClassData()));
		DataSetsIterator itTest = testSets.find(it->first);
		itTest->second.reserve((1-facForTraining) * it->second.size());
		for(int i = 0; i < it->second.size(); ++i){
			if(i <= min(300, (int) (facForTraining * it->second.size()))){
				itTrain->second.push_back(it->second[i]);
			}else{
				itTest->second.push_back(it->second[i]);
			}
		}
	}

	for(DataSets::const_iterator it = trainSets.begin(); it != trainSets.end(); ++it){
		std::cout << "for training: "<< it->first << ", has: " << it->second.size() << std::endl;
	}
	for(DataSets::const_iterator it = testSets.begin(); it != testSets.end(); ++it){
		std::cout << "for testing:  "<< it->first << ", has: " << it->second.size() << std::endl;
	}

	int height;
	int amountOfTrees;
	Settings::getValue("Forest.Trees.height", height, 7);
	Settings::getValue("Forest.amountOfTrees", amountOfTrees, 1000);
//#define WRITE
#ifdef WRITE
	RandomForestGaussianProcess rfGp(trainSets, height, amountOfTrees, path);
	rfGp.train();
	RFGPWriter::writeToFile("rfGp.rfgpbin", rfGp);
#else
	RandomForestGaussianProcess rfGp(trainSets);
	RFGPWriter::readFromFile("rfGp.rfgpbin", rfGp);
#endif
	std::cout << CYAN << "Finish training -> Start prediction" << RESET << std::endl;
	int labelCounter = 0;
	int correct = 0;
	int amount = 0;
	std::vector<double> prob;
	for(DataSets::const_iterator it = testSets.begin(); it != testSets.end(); ++it){
		for(int i = 0; i < it->second.size(); ++i){
			const int rfGPLabel = rfGp.predict(*it->second[i], prob);
			//std::cout << "Should: " << rfGPLabel << ", is: " << labelCounter << std::endl;
			if(rfGPLabel == labelCounter){
				++correct;
			}
			++amount;
		}
		++labelCounter;
	}
	std::cout << RED << "Amount of test data: " << amount << RESET << std::endl;
	std::cout << RED << "Amount of right: " << (double) correct / amount * 100.0 << "%" << RESET << std::endl;

	std::cout << CYAN << "Start second prediction" << RESET << std::endl;
	labelCounter = 0;
	correct = 0;
	amount = 0;
	for(DataSets::const_iterator it = trainSets.begin(); it != trainSets.end(); ++it){
		for(int i = 0; i < it->second.size(); ++i){
			const int rfGPLabel = rfGp.predict(*it->second[i], prob);
			//std::cout << "Should: " << rfGPLabel << ", is: " << labelCounter << std::endl;
			if(rfGPLabel == labelCounter){
				++correct;
			}
			++amount;
		}
		++labelCounter;
	}
	std::cout << RED << "Amount of test data: " << amount << RESET << std::endl;
	std::cout << RED << "Amount of right: " << (double) correct / amount * 100.0 << "%" << RESET << std::endl;

/*RandomForestGaussianProcess rfGp(trainSets);
	RFGPWriter::readFromFile("rfGp.rfgpbin", rfGp);
	*/

//	DataWriterForVisu::generateGrid("out.txt", rfGp, 40, data, 0, 1);

	std::cout << "finish" << std::endl;
	return;
}

#endif /* TESTS_MULTICLASSRFGPTEST_H_ */
