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
#include "RandomForests/DecisionTree.h"

int main(){
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



	DecisionTree tree(4);
	Data data;
	Eigen::VectorXd first(5);
	first << 1,2,4,5,6;
	Eigen::VectorXd second(5);
	second << 2,3,1,2,7;
	data.push_back(first);
	data.push_back(second);
	Labels labels;
	labels.push_back(0);
	labels.push_back(1);
	Eigen::Vector2i minMaxUsedData;
	minMaxUsedData << (int) 0.3 * data.size(), (int) 0.7 * data.size();
	tree.train(data, labels, 5, minMaxUsedData);
	return 0;
}


