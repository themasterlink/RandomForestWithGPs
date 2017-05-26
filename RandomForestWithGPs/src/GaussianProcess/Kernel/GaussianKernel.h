/*
 * GaussianKernel.h
 *
 *  Created on: 01.07.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_KERNEL_H_
#define GAUSSIANPROCESS_KERNEL_H_

#include "../../Data/LabeledVectorX.h"
#include "../../RandomNumberGenerator/RandomGaussianNr.h"
#include "KernelType.h"
#include "KernelBase.h"

class GaussianProcessWriter;


class GaussianKernel : public KernelBase<GaussianKernelParams> {
	friend GaussianProcessWriter;
public:

	GaussianKernel(bool simpleLength = true);
	virtual ~GaussianKernel();

	bool hasLengthMoreThanOneDim() const{ return m_kernelParams.m_length.hasMoreThanOneDim(); };

	void changeKernelConfig(const bool useAllDimForLen);

	double calcDiagElement(unsigned int row) const override;

	double calcDerivativeDiagElement(unsigned int row, const OwnKernelElement* type) const override;

	void calcKernelVector(const VectorX& vector, const Matrix& dataMat, VectorX& res) const override;

	void calcCovariance(Matrix& cov) const;

	void calcCovarianceDerivative(Matrix& cov, const OwnKernelElement* type) const;

	void calcCovarianceDerivativeForInducingPoints(Matrix& cov, const std::list<unsigned int>& activeSet, const OwnKernelElement* type, const int element = -1) const;

	std::string prettyString() const override;

	void setHyperParams(double len, double noiseF);

	void setHyperParams(double len, double noiseF, double noiseS);

	void setHyperParams(const std::vector<real>& len, double noiseF, double noiseS);

	void addHyperParams(double len, double noiseF, double noiseS = 0.);

	void setSNoise(double val){ m_kernelParams.m_sNoise.setAllValuesTo(val); };

	void setFNoise(double val){ m_kernelParams.m_fNoise.setAllValuesTo(val); };

	void setLength(double val){ m_kernelParams.m_length.setAllValuesTo(val); };

	void getCopyOfParams(GaussianKernelParams& params);

	double kernelFunc(const int row, const int col) const override;

	double kernelFuncVec(const VectorX& lhs, const VectorX& rhs) const override;

	double kernelFuncDerivativeToParam(const int row, const int col, const OwnKernelElement* type, const int element = -1) const override;

private:

	bool kernelCanHaveDifferenceMatrix() const override {return true;};

};

inline
double GaussianKernel::calcDiagElement(unsigned int row) const{ // row is not used in this kernel!
	UNUSED(row);
	return m_kernelParams.m_fNoise.getSquaredValue() + m_kernelParams.m_sNoise.getSquaredValue();
}

#endif /* GAUSSIANPROCESS_KERNEL_H_ */
