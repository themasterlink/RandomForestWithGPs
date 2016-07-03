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
#include "Kernel.h"

class GaussianProcessBinaryClass{
public:

	enum Status {
		NANORINFERROR = -1,
		ALLFINE = 0
	};

	GaussianProcessBinaryClass();
	virtual ~GaussianProcessBinaryClass();

	void init(const Eigen::MatrixXd& dataMat, const Eigen::VectorXd& y);

	void train();

	double predict(const DataElement& newPoint) const;

private:
	Status trainF(const int dataPoints, const Eigen::MatrixXd& K, const Eigen::VectorXd& y);

	Status trainLM(double& logZ, std::vector<double>& dLogZ);

	Status train(const int dataPoints, const Eigen::MatrixXd& dataMat, const Eigen::VectorXd& y);

	void updatePis(const int dataPoints, const Eigen::VectorXd& y, const Eigen::VectorXd& t);

	double m_repetitionStepFactor;

	Eigen::MatrixXd m_dataMat;
	Eigen::VectorXd m_a;
	Eigen::VectorXd m_y;
	Eigen::VectorXd m_f;
	Eigen::VectorXd m_pi;
	Eigen::VectorXd m_dLogPi;
	Eigen::VectorXd m_ddLogPi;
	Eigen::VectorXd m_sqrtDDLogPi;
	Eigen::LLT<Eigen::MatrixXd> m_choleskyLLT;
	int m_dataPoints; // amount of data points
	StopWatch m_sw;

	Kernel m_kernel;
	bool m_init;
	bool m_trained;
};

#endif /* GAUSSIANPROCESS_GAUSSIANPROCESSBINARYCLASS_H_ */
