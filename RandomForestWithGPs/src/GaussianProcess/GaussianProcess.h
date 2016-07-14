/*
 * GaussianProcessBinaryClass.h
 *
 *  Created on: 15.06.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_GAUSSIANPROCESS_H_
#define GAUSSIANPROCESS_GAUSSIANPROCESS_H_

#include "../Utility/Util.h"
#include <cmath>
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include "Kernel.h"

class GaussianProcessWriter;
class BayesOptimizer;

class GaussianProcess{
	friend GaussianProcessWriter;
	friend BayesOptimizer;
public:

	enum Status {
		NANORINFERROR = -1,
		ALLFINE = 0
	};

	GaussianProcess();
	virtual ~GaussianProcess();

	void init(const Eigen::MatrixXd& dataMat, const Eigen::VectorXd& y);

	void train();

	void trainWithoutKernelOptimize();

	double predict(const DataElement& newPoint) const;

	Kernel& getKernel(){ return m_kernel; };

	double getLenMean() const {return m_kernel.getLenMean();};

private:
	GaussianProcess::Status trainBayOpt(double& logZ, const double lambda);

	Status trainLM(double& logZ, std::vector<double>& dLogZ);

	Status trainF(const int dataPoints, const Eigen::MatrixXd& K, const Eigen::VectorXd& y);

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
	Eigen::MatrixXd m_innerOfLLT;
	Eigen::LLT<Eigen::MatrixXd> m_choleskyLLT;
	int m_dataPoints; // amount of data points
	StopWatch m_sw;

	bool m_init;
	bool m_trained;

	Kernel m_kernel;
};

#endif /* GAUSSIANPROCESS_GAUSSIANPROCESS_H_ */
