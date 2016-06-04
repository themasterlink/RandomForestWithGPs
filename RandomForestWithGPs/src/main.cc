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

// just for testing
void write(const std::string& fileName, const OtherRandomForest& forest, const double amountOfPointsOnOneAxis, const Eigen::Vector2d& min, const Eigen::Vector2d& max){
	Eigen::Vector2d stepSize = (1./amountOfPointsOnOneAxis) * (max - min);
	std::ofstream file;
	file.open(fileName);
	Data points; points.reserve(amountOfPointsOnOneAxis*(amountOfPointsOnOneAxis+1));
	int amount = 0;
	for(double x = max[0]; x >= min[0] ; x -= stepSize[0]){
		for(double y = min[1]; y < max[1]; y += stepSize[1]){
			DataElement ele(2);
			ele << x,y;
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

	/*
	Eigen::MatrixXd xA(10,1);
	Eigen::VectorXd col; // input data
	xA(0,0) = 0;
	xA(1,0) = 0.1;
	xA(2,0) = 0.2;
	xA(3,0) =  0.3;
	xA(4,0) = 0.6;
	xA(5,0) =  1.;
	xA(6,0) =  1.1;
	xA(7,0) =  1.5;
	xA(8,0) =  1.9;
	xA(9,0) =  2.0;
	Eigen::MatrixXd res;
	Kernel::getSeKernelFor(xA, xA, res, 5.0, 0.2);

	std::cout << "res: " << res << std::endl;
	Eigen::MatrixXd inv = res.inverse();
	std::cout << "inv: " << inv << std::endl;
	 */

	Data data;
	Labels labels;
	DataReader::readTrainingFromFile(data, labels, "../testData/testInput2.txt");

	Eigen::Vector2i minMaxUsedData;
	minMaxUsedData << (int) (0.2 * data.size()), (int) (0.6 * data.size());


	Data testData;
	Labels testLabels;
	DataReader::readTrainingFromFile(testData, testLabels, "../testData/testInput3.txt");

	std::cout << "Finished reading" << std::endl;

	OtherRandomForest otherForest(7,1000,2);
	otherForest.train(data, labels, 2, minMaxUsedData);


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

	StopWatch sw;
	write("../testData/trainedResult2.txt", otherForest, 200, min, max);
	std::cout << "Time for write: " << sw.elapsedSeconds() << std::endl;
	std::cout << "End Reached" << std::endl;
	return 0;
}


