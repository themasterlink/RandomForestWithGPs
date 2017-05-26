/*
 * performanceMeasurement.h
 *
 *  Created on: 14.09.2016
 *      Author: Max
 */

#ifndef TESTS_PERFORMANCEMEASUREMENT_H_
#define TESTS_PERFORMANCEMEASUREMENT_H_

#include "../Utility/Util.h"

bool testSpeedOfEigenMultWithDiag(){
	const int dataPoints = 300;
	Matrix innerOfLLT(Matrix::Identity(dataPoints,dataPoints));
	Matrix innerOfLLT2(Matrix::Identity(dataPoints,dataPoints));
	VectorX sqrtDDLogPi(VectorX::Random(dataPoints));
	const Matrix K(Matrix::Random(dataPoints,dataPoints));
	StopWatch sw;
	for(int i = 0; i < 1; ++i){
		{
			const Matrix eye(Matrix::Identity(dataPoints,dataPoints));
			const Matrix WSqrt( DiagMatrixXd(sqrtDDLogPi).toDenseMatrix()); // TODO more efficient

			//std::cout << "K: \n" << K << std::endl;
			//std::cout << "inner: \n" << eye + (WSqrt * K * WSqrt) << std::endl;
			innerOfLLT = eye + (WSqrt * K * WSqrt);
		}
	}
	std::cout << "Usual way takes: " << sw.elapsedAsPrettyTime() << std::endl;

	StopWatch sw2;
	for(int k = 0; k < 1; ++k){
		{
			for(int i = 0; i < dataPoints; ++i){
				for(int j = 0; j < dataPoints; ++j){
					innerOfLLT2(i,j) = sqrtDDLogPi[i] * sqrtDDLogPi[j] * K(i,j);
				}
				innerOfLLT2(i,i) += 1;
			}
		}
	}

	std::cout << "New way takes: " << sw2.elapsedAsPrettyTime() << std::endl;
	for(int i = 0; i < dataPoints; ++i){
		for(int j = 0; j < dataPoints; ++j){
			if(fabs(innerOfLLT2(i,j) - innerOfLLT(i,j)) <= EPSILON){
				return false;
			}
		}
	}
	return true;
}


#endif /* TESTS_PERFORMANCEMEASUREMENT_H_ */
