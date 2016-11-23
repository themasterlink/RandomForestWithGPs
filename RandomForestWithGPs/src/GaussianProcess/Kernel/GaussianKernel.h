/*
 * GaussianKernel.h
 *
 *  Created on: 01.07.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_KERNEL_H_
#define GAUSSIANPROCESS_KERNEL_H_

#include "../../Data/ClassData.h"
#include "../../RandomNumberGenerator/RandomGaussianNr.h"
#include "KernelType.h"
#include "KernelBase.h"
#include <list>

class GaussianProcessWriter;


class GaussianKernel : public KernelBase<GaussianKernelParams> {
	friend GaussianProcessWriter;
public:

	GaussianKernel(bool simpleLength = true);
	virtual ~GaussianKernel();

	const bool hasLengthMoreThanOneDim() const{ return m_kernelParams.m_length.hasMoreThanOneDim(); };

	void changeKernelConfig(const bool useAllDimForLen);

	double calcDiagElement(unsigned int row) const;

	double calcDerivativeDiagElement(unsigned int row, const OwnKernelElement* type) const;

	void calcKernelVector(const Eigen::VectorXd& vector, const Eigen::MatrixXd& dataMat, Eigen::VectorXd& res) const;

	void calcCovariance(Eigen::MatrixXd& cov) const;

	void calcCovarianceDerivative(Eigen::MatrixXd& cov, const OwnKernelElement* type) const;

	void calcCovarianceDerivativeForInducingPoints(Eigen::MatrixXd& cov, const std::list<int>& activeSet, const OwnKernelElement* type, const int element = -1) const;

	std::string prettyString() const;

	void setHyperParams(double len, double noiseF);

	void setHyperParams(double len, double noiseF, double noiseS);

	void setHyperParams(const std::vector<double>& len, double noiseF, double noiseS);

	void addHyperParams(double len, double noiseF, double noiseS = 0.);

	void setSNoise(double val){ m_kernelParams.m_sNoise.setAllValuesTo(val); };

	void setFNoise(double val){ m_kernelParams.m_fNoise.setAllValuesTo(val); };

	void setLength(double val){ m_kernelParams.m_length.setAllValuesTo(val); };

	void getCopyOfParams(GaussianKernelParams& params);

	double kernelFunc(const int row, const int col) const;

	double kernelFuncVec(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const;

	double kernelFuncDerivativeToParam(const int row, const int col, const OwnKernelElement* type, const int element = -1) const;

};

inline
double GaussianKernel::calcDiagElement(unsigned int row) const{ // row is not used in this kernel!
	return m_kernelParams.m_fNoise.getSquaredValue() + m_kernelParams.m_sNoise.getSquaredValue();
}

#endif /* GAUSSIANPROCESS_KERNEL_H_ */
