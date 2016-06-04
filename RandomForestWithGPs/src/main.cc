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
#include "RandomForests/RandomForest.h"
#include "RandomForests/OtherRandomForest.h"
#include "Utility/Settings.h"
#include "Data/DataReader.h"
#include "Data/DataWriterForVisu.h"

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

	Eigen::Vector2i minMaxUsedData;
	minMaxUsedData << (int) (0.2 * data.size()), (int) (0.6 * data.size());

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

	OtherRandomForest otherForest(height, amountOfTrees, dim);
	otherForest.train(data, labels, dim, minMaxUsedData);

	Labels guessedLabels;
	otherForest.predictData(testData, guessedLabels);

	int wrong = 0;
	for(int i = 0; i < testData.size(); ++i){
		if(guessedLabels[i] != testLabels[i]){
			++wrong;
		}
	}
	std::cout << "Other: Amount of test size: " << testData.size() << std::endl;
	std::cout << "Other: Amount of wrong: " << wrong / (double) testData.size() << std::endl;

	/*RandomForest forest(9,500,2);
	 //forest.train(data, labels, 2, minMaxUsedData);
	 wrong = 0;
	 for(int i = 0; i < testData.size(); ++i){
	 int label = forest.predict(testData[i]);
	 if(label != testLabels[i]){
	 //std::cout << "Label wrong for : " << i << ", " << label << " = " << testLabels[i] << std::endl;
	 ++wrong;
	 }
	 }
	 std::cout << "Amount of test size: " << testData.size() << std::endl;
	 std::cout << "Amount of wrong: " << wrong / (double) testData.size() << std::endl;
	 */


	bool doWriting;
	int printX, printY;
	Settings::getValue("Write2D.doWriting", doWriting, false);
	Settings::getValue("Write2D.printX", printX, 0);
	Settings::getValue("Write2D.printY", printY, 1);
	if(doWriting){
		Settings::getValue("Write2D.gridPath", path);
		StopWatch sw;
		DataWriterForVisu::generateGrid(path, otherForest, 200, data, printX, printY);
		Settings::getValue("Write2D.testPath", path);
		DataWriterForVisu::writeData(path, testData, testLabels, printX, printY);
		std::cout << "Time for write: " << sw.elapsedSeconds() << std::endl;
		std::cout << "End Reached" << std::endl;
		system("../PythonScripts/plotData.py");
	}
	return 0;
}

