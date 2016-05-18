/*
 * kernelCalc.h
 *
 *  Created on: 18.05.2016
 *      Author: Max
 */

#ifndef KERNELCALC_H_
#define KERNELCALC_H_

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




#endif /* KERNELCALC_H_ */
