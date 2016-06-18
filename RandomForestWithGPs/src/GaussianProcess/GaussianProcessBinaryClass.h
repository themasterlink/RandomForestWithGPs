/*
 * GaussianProcessBinaryClass.h
 *
 *  Created on: 15.06.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_GAUSSIANPROCESSBINARYCLASS_H_
#define GAUSSIANPROCESS_GAUSSIANPROCESSBINARYCLASS_H_

#include "../Utility/Util.h"
#include <cmath>
#include <Eigen/Cholesky>
#include <Eigen/Dense>


class GaussianProcessBinaryClass{
public:
	GaussianProcessBinaryClass();
	virtual ~GaussianProcessBinaryClass();

	// todo move to a better place
	typedef Eigen::DiagonalWrapper<const Eigen::MatrixXd> DiagMatrixXd;

	void train(const int dataPoints, const Eigen::MatrixXd& K, const Eigen::VectorXd& y);

	double predict(const Eigen::VectorXd newPoint);


	Eigen::MatrixXd m_dataMat;
private:
	Eigen::VectorXd m_f;
	Eigen::VectorXd m_pi;
	Eigen::VectorXd m_dLogPi;
	Eigen::VectorXd m_ddLogPi;
	Eigen::VectorXd m_sqrtDDLogPi;
	Eigen::MatrixXd m_lowerOfCholesky;

};

#endif /* GAUSSIANPROCESS_GAUSSIANPROCESSBINARYCLASS_H_ */
