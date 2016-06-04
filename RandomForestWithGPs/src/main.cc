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

// just for testing
void write(const std::string& fileName, const OtherRandomForest& forest,
		const double amountOfPointsOnOneAxis, const Eigen::Vector2d& min,
		const Eigen::Vector2d& max){
	Eigen::Vector2d stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	std::ofstream file;
	file.open(fileName);
	Data points;
	points.reserve(amountOfPointsOnOneAxis * (amountOfPointsOnOneAxis + 1));
	int amount = 0;
	for(double x = max[0]; x >= min[0]; x -= stepSize[0]){
		for(double y = min[1]; y < max[1]; y += stepSize[1]){
			DataElement ele(2);
			ele << x, y;
			points.push_back(ele);
			++amount;
		}
	}
	Labels labels;
	forest.predictData(points, labels);
	for(int i = 0; i < amount; ++i){
		file << points[i][0] << " " << points[i][1] << " " << labels[i] << "\n";
	}
	file.close();
}

int main(){

	std::cout << "Start" << std::endl;

	// read in Settings
	Settings::init("../Settings/init.json");
	Data data;
	Labels labels;
	std::string path;
	Settings::getValue("Training.path", path);
	DataReader::readTrainingFromFile(data, labels, path);

	Eigen::Vector2i minMaxUsedData;
	minMaxUsedData << (int) (0.2 * data.size()), (int) (0.6 * data.size());

	Data testData;
	Labels testLabels;
	Settings::getValue("Test.path", path);
	DataReader::readTrainingFromFile(testData, testLabels, path);

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

	Eigen::Vector2d min, max;
	min[0] = min[1] = 1000000;
	max[0] = max[1] = -1000000;
	for(int i = 0; i < data.size(); ++i){
		for(int j = 0; j < data[i].rows(); ++j){
			if(min[j] > data[i][j]){
				min[j] = data[i][j];
			}
			if(max[j] < data[i][j]){
				max[j] = data[i][j];
			}
		}
	}
	bool doWriting;
	Settings::getValue("Write2D.doWriting", doWriting, false);
	if(doWriting){
		Settings::getValue("Write2D.path", path);
		StopWatch sw;
		write(path, otherForest, 200, min, max);
		std::cout << "Time for write: " << sw.elapsedSeconds() << std::endl;
		std::cout << "End Reached" << std::endl;
	}
	return 0;
}

