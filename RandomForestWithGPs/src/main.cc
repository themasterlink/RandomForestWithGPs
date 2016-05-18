//============================================================================
// Name        : RandomForestWithGPs.cpp
// Author      : 
// Version     :
// Copyright   : 
// Description :
//============================================================================

#include <iostream>
#include <Eigen/Dense>
#include <math.h>

void getSeKernelFor(const Eigen::MatrixXd& xA, const Eigen::MatrixXd& xB, Eigen::MatrixXd& result, const double sigma = 1.0, const double l = 0.1){
	result = Eigen::MatrixXd(xA.rows(), xB.rows());
	const double sigmaSquared = sigma * sigma;
	const double expFac = -1. / (2.0 * l * l);
	for(int i = 0; i < xA.rows(); ++i){
		const Eigen::VectorXd xARow = xA.row(i);
		for(int j = 0; j < xB.rows(); ++j){
			const Eigen::VectorXd diff = xARow - xB.row(j);
			result(i,j) = sigmaSquared * exp(expFac * diff.dot(diff));
		}
	}
}

int main(){
	Eigen::MatrixXd xA(10,1);
	Eigen::VectorXd col;
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
	getSeKernelFor(xA, xA, res, 5.0, 0.2);

	std::cout << "res: " << res << std::endl;
	Eigen::MatrixXd inv = res.inverse();
	std::cout << "inv: " << inv << std::endl;
	//int main() {
	std::cout << "Hallo" << std::endl; // prints
	return 0;
}


