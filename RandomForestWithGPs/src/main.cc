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

// just for testing
void write(const std::string& fileName, const RandomForest& forest, const double amountOfPointsOnOneAxis, const Eigen::Vector2d& min, const Eigen::Vector2d& max){
	Eigen::Vector2d stepSize = (1./amountOfPointsOnOneAxis) * (max - min);
	std::ofstream file;
	file.open(fileName);
	for(double x = min[0]; x < max[0]; x += stepSize[0]){
		for(double y = min[1]; y < max[1]; y += stepSize[1]){
			DataElement ele(2);
			ele << x,y;
			const int label = forest.predict(ele);
			file << x << "," << y << "," << label << std::endl;
		}
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

	RandomForest forest(7,5000,2);

	//DecisionTree tree(6, 2);

	Eigen::Vector2i minMaxUsedData;
	minMaxUsedData << (int) ( data.size()), (int) data.size();
	forest.train(data, labels, 2, minMaxUsedData);

	Data testData;
	Labels testLabels;
	DataReader::readTrainingFromFile(testData, testLabels, "../testData/testInput3.txt");
	int wrong = 0;
	for(int i = 0; i < testData.size(); ++i){
		int label = forest.predict(testData[i]);
		if(label != testLabels[i]){
			//std::cout << "Label wrong for : " << i << ", " << label << " = " << testLabels[i] << std::endl;
			++wrong;
		}
	}
	std::cout << "Amount of test size: " << testData.size() << std::endl;
 	std::cout << "Amount of wrong: " << wrong / (double) testData.size() << std::endl;

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

	write("../testData/trainedResult2.txt", forest, 80, min, max);

	std::cout << "End Reached" << std::endl;
	return 0;
}


