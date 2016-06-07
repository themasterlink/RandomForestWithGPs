//============================================================================
// Name        : RandomForestWithGPs.cpp
// Author      : 
// Version     :
// Copyright   : 
// Description :
//============================================================================

#include <iostream>
#include <Eigen/Dense>
#include "kernelCalc.h"
#include "Utility/Settings.h"
#include "Data/DataReader.h"
#include "Data/DataWriterForVisu.h"
#include "RandomForests/RandomForest.h"
#include "RandomForests/RandomForestWriter.h"

// just for testing


int main(){

	std::cout << "Start" << std::endl;

	// read in Settings
	Settings::init("../Settings/init.json");
	Data data;
	Labels labels;
	std::string path;
	Settings::getValue("Training.path", path);
	DataReader::readFromFile(data, labels, path);

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

	Data testData;
	Labels testLabels;
	Settings::getValue("Test.path", path);
	DataReader::readFromFile(testData, testLabels, path);

	std::cout << "Finished reading" << std::endl;
	int dim = 2;
	if(data.size() > 0){
		dim = data[0].rows();
	}

	int height;
	int amountOfTrees;
	Settings::getValue("Forest.Trees.height", height, 7);
	Settings::getValue("Forest.amountOfTrees", amountOfTrees, 1000);
	std::cout << "Amount of trees: " << amountOfTrees << " with height: " << height << std::endl;

	RandomForest forest(height, amountOfTrees, dim);
	forest.train(data, labels, dim, minMaxUsedData);

	Labels guessedLabels;
	forest.predictData(testData, guessedLabels);

	int wrong = 0;
	for(int i = 0; i < testData.size(); ++i){
		if(guessedLabels[i] != testLabels[i]){
			++wrong;
		}
	}
	std::cout << "Other: Amount of test size: " << testData.size() << std::endl;
	std::cout << "Other: Amount of wrong: " << wrong / (double) testData.size() << std::endl;



	bool doWriting;
	Settings::getValue("WriteBinarySaveOfTrees.doWriting", doWriting, false);
	if(doWriting){
		std::string path;
		Settings::getValue("WriteBinarySaveOfTrees.path", path);
		RandomForestWriter::writeToFile(path, forest);
	}
	RandomForest newForest(0,0,0);
	RandomForestWriter::readFromFile("../testData/trees2.binary", newForest);
	Labels guessedLabels2;
	newForest.addForest(forest);
	newForest.predictData(testData, guessedLabels2);

	wrong = 0;
	for(int i = 0; i < testData.size(); ++i){
		if(guessedLabels2[i] != testLabels[i]){
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
		Settings::getValue("Write2D.gridPath", path);
		StopWatch sw;
		DataWriterForVisu::generateGrid(path, forest, 200, data, printX, printY);
		Settings::getValue("Write2D.testPath", path);
		DataWriterForVisu::writeData(path, testData, testLabels, printX, printY);
		std::cout << "Time for write: " << sw.elapsedSeconds() << std::endl;
		std::cout << "End Reached" << std::endl;
		system("../PythonScripts/plotData.py");
	}



	return 0;
}

