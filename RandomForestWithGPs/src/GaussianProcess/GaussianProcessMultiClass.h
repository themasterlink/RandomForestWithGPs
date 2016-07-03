/*
 * GaussianProcessMultiClass.h
 *
 *  Created on: 15.06.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_GAUSSIANPROCESSMULTICLASS_H_
#define GAUSSIANPROCESS_GAUSSIANPROCESSMULTICLASS_H_

#include "../Utility/Util.h"
#include "../Data/Data.h"
#include <cmath>
#include <Eigen/Cholesky>
#include <Eigen/Dense>


class GaussianProcessMultiClass{
public:

	static void calcPhiBasedOnF(const Eigen::VectorXd& f, Eigen::VectorXd& pi, const int amountOfClasses, const int dataPoints);

	static void magicFunc(const int amountOfClasses, const int dataPoints, const std::vector<Eigen::MatrixXd>& K_c, const Eigen::VectorXd& y);

private:
	GaussianProcessMultiClass();
	virtual ~GaussianProcessMultiClass();
};

#endif /* GAUSSIANPROCESS_GAUSSIANPROCESSMULTICLASS_H_ */
