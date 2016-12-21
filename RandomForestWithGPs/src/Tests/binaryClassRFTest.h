/*
 * binaryClassRFTest.h
 *
 *  Created on: 15.09.2016
 *      Author: Max
 */

#ifndef TESTS_BINARYCLASSRFTEST_H_
#define TESTS_BINARYCLASSRFTEST_H_

#include <Eigen/Dense>
#include "../Data/DataReader.h"
#include "../Data/DataWriterForVisu.h"
#include "../RandomForests/RandomForest.h"
#include "../RandomForests/RandomForestWriter.h"
#include "../Base/Settings.h"

void executeForRFBinaryClass(){
	ClassData data;
	ClassData testData;
	DataSets datas;
	bool didNormalize = false;
	DataReader::readFromFiles(datas, "../realData/", 500, false, didNormalize);
	if(!didNormalize){
		DataPoint center, var;
		DataConverter::centerAndNormalizeData(datas, center, var);
	}
	int labelCounter = 0;
	for(DataSetsIterator itData = datas.begin(); itData != datas.end(); ++itData){
		const int amountOfElements = itData->second.size();
		for(int i = 0; i < amountOfElements; ++i){
			if(i <= min(300, (int) (0.8 * amountOfElements))){
				// train data
				data.push_back(itData->second[i]);
			}else{
				// test data
				testData.push_back(itData->second[i]);
			}
		}
		++labelCounter;
	}

	std::cout << "Training size: " << data.size() << std::endl;
	std::cout << "Test size:     " << testData.size() << std::endl;
	std::vector<int> heights(5);
	heights[0] = 2;
	heights[1] = 4;
	heights[2] = 6;
	heights[3] = 8;
	heights[4] = 12;
	std::vector<int> trees(5);
	trees[0] = 100;
	trees[1] = 1000;
	trees[2] = 10000;
	trees[3] = 100000;
	trees[4] = 1000000;

	for(int i = 0; i < heights.size(); ++i){
		int maxTree = trees.size();
		if(heights[i] > 10){
			maxTree = 3;
		}else if(heights[i] > 7){
			maxTree = 4;
		}
		for(int j = 0; j < maxTree; ++j){
//			, 6, 1000
			const int height = 6; //heights[i];
			const int amountOfTrees = 1000; //trees[j];
			// for binary case:
			//const int dataPoints = data.size();
			std::cout << "Amount of trees: " << amountOfTrees << " with height: " << height << std::endl;
			bool useFixedValuesForMinMaxUsedData;
			Settings::getValue("MinMaxUsedData.useFixedValuesForMinMaxUsedData", useFixedValuesForMinMaxUsedData);
			Eigen::Vector2i minMaxUsedData;
			if(useFixedValuesForMinMaxUsedData){
				int minVal = 0, maxVal = 0;
				Settings::getValue("MinMaxUsedData.minValue", minVal);
				Settings::getValue("MinMaxUsedData.maxValue", maxVal);
				minMaxUsedData << minVal, maxVal;
			}else{
				double minVal = 0, maxVal = 0;
				Settings::getValue("MinMaxUsedData.minValueFraction", minVal);
				Settings::getValue("MinMaxUsedData.maxValueFraction", maxVal);
				minMaxUsedData << (int) (minVal * data.size()),  (int) (maxVal * data.size());
			}
			std::cout << "Min max used data, min: " << minMaxUsedData[0] << " max: " << minMaxUsedData[1] << "\n";

			RandomForest forest(height, amountOfTrees, data[0]->rows());
			forest.train(data, data[0]->rows(), minMaxUsedData);

			int right = 0;
			Labels predictedLabels;
			forest.predictData(testData, predictedLabels);
			for(int j = 0; j < testData.size(); ++j){
				if(testData[j]->getLabel() == predictedLabels[j]){
					++right;
				}
			}
			std::cout << RED << "Amount of right: " << (double) right / testData.size() * 100.0 << "%" << RESET << std::endl;
			break;
		}
		break;
	}
}

void executeForRFBinaryClass(const std::string& path){
	ClassData data;
	DataReader::readFromFile(data, path, 500);
	bool useFixedValuesForMinMaxUsedData;
	Settings::getValue("MinMaxUsedData.useFixedValuesForMinMaxUsedData", useFixedValuesForMinMaxUsedData);
	Eigen::Vector2i minMaxUsedData;
	if(useFixedValuesForMinMaxUsedData){
		int minVal = 0, maxVal = 0;
		Settings::getValue("MinMaxUsedData.minValue", minVal);
		Settings::getValue("MinMaxUsedData.maxValue", maxVal);
		minMaxUsedData << minVal, maxVal;
	}else{
		double minVal = 0, maxVal = 0;
		Settings::getValue("MinMaxUsedData.minValueFraction", minVal);
		Settings::getValue("MinMaxUsedData.maxValueFraction", maxVal);
		minMaxUsedData << (int) (minVal * data.size()),  (int) (maxVal * data.size());
	}
	std::cout << "Min max used data, min: " << minMaxUsedData[0] << " max: " << minMaxUsedData[1] << "\n";

	ClassData testData;
	std::string testPath;
	Settings::getValue("Test.path", testPath);
	DataReader::readFromFile(testData, testPath, 500);

	std::cout << "Finished reading" << std::endl;
	int dim = 2;
	if(data.size() > 0){
		dim = data[0]->rows();
	}
	int amountOfTrees, height;
	Settings::getValue("Forest.amountOfTrees", amountOfTrees);
	Settings::getValue("Forest.Trees.height", height);
	std::cout << "Amount of trees: " << amountOfTrees << " with height: " << height << std::endl;

	RandomForest forest(height, amountOfTrees, dim);
	forest.train(data, dim, minMaxUsedData);

	Labels guessedLabels;
	forest.predictData(testData, guessedLabels);

	int wrong = 0;
	for(int i = 0; i < testData.size(); ++i){
		if(guessedLabels[i] != testData[i]->getLabel()){
			++wrong;
		}
	}
	std::cout << "Other: Amount of test size: " << testData.size() << std::endl;
	std::cout << "Other: Amount of wrong: " << wrong / (double) testData.size() << std::endl;



	bool doWriting;
	Settings::getValue("WriteBinarySaveOfTrees.doWriting", doWriting, false);
	std::string writePath;
	if(doWriting){
		Settings::getValue("WriteBinarySaveOfTrees.path", writePath);
		RandomForestWriter::writeToFile(writePath, forest);
	}
	RandomForest newForest(0,0,0);
	RandomForestWriter::readFromFile("../testData/trees2.binary", newForest);
	Labels guessedLabels2;
	newForest.addForest(forest);
	newForest.predictData(testData, guessedLabels2);

	wrong = 0;
	for(int i = 0; i < testData.size(); ++i){
		if(guessedLabels2[i] != testData[i]->getLabel()){
			++wrong;
		}
	}
	std::cout << "Amount of combined trees: " << newForest.getNrOfTrees() << std::endl;
	std::cout << "Read: Amount of test size: " << testData.size() << std::endl;
	std::cout << "Read: Amount of wrong: " << wrong / (double) testData.size() << std::endl;


	int printX, printY;
	Settings::getValue("Write2D.doWriting", doWriting, false);
	Settings::getValue("Write2D.printX", printX, 0);
	Settings::getValue("Write2D.printY", printY, 1);
	if(doWriting){
		Settings::getValue("Write2D.gridPath", writePath);
		StopWatch sw;
		DataWriterForVisu::generateGrid(writePath, &forest, 200, data, printX, printY);
		Settings::getValue("Write2D.testPath", writePath);
		DataWriterForVisu::writeData(writePath, testData, printX, printY);
		std::cout << "Time for write: " << sw.elapsedSeconds() << std::endl;
		std::cout << "End Reached" << std::endl;
		system("../PythonScripts/plotData.py");
	}
}

#endif /* TESTS_BINARYCLASSRFTEST_H_ */
