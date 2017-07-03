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

#ifdef BUILD_OLD_CODE
class GaussianProcessWriter;
#endif // BUILD_OLD_CODE

class GaussianKernel : public KernelBase<GaussianKernelParams> {
#ifdef BUILD_OLD_CODE
	friend GaussianProcessWriter;
#endif // BUILD_OLD_CODE
public:

	GaussianKernel(bool simpleLength = true);
	virtual ~GaussianKernel();

	bool hasLengthMoreThanOneDim() const{ return m_kernelParams.m_length.hasMoreThanOneDim(); };

	void changeKernelConfig(const bool useAllDimForLen);

	Real calcDiagElement(unsigned int row) const override;

	Real calcDerivativeDiagElement(unsigned int row, const OwnKernelElement* type) const override;

	void calcKernelVector(const VectorX& vector, const Matrix& dataMat, VectorX& res) const override;

	void calcCovariance(Matrix& cov) const;

	void calcCovarianceDerivative(Matrix& cov, const OwnKernelElement* type) const;

	void calcCovarianceDerivativeForInducingPoints(Matrix& cov, const std::list<unsigned int>& activeSet, const OwnKernelElement* type, const int element = -1) const;

	std::string prettyString() const override;

	void setHyperParams(Real len, Real noiseF);

	void setHyperParams(Real len, Real noiseF, Real noiseS);

	void setHyperParams(const std::vector<Real>& len, Real noiseF, Real noiseS);

	void addHyperParams(Real len, Real noiseF, Real noiseS = 0.);

	void setSNoise(Real val){ m_kernelParams.m_sNoise.setAllValuesTo(val); };

	void setFNoise(Real val){ m_kernelParams.m_fNoise.setAllValuesTo(val); };

	void setLength(Real val){ m_kernelParams.m_length.setAllValuesTo(val); };

	void getCopyOfParams(GaussianKernelParams& params);

	Real kernelFunc(const int row, const int col) const override;

	Real kernelFuncVec(const VectorX& lhs, const VectorX& rhs) const override;

	Real kernelFuncDerivativeToParam(const int row, const int col, const OwnKernelElement* type, const int element = -1) const override;

private:

	bool kernelCanHaveDifferenceMatrix() const override {return true;};

};

inline
Real GaussianKernel::calcDiagElement(unsigned int row) const{ // row is not used in this kernel!
	UNUSED(row);
	return m_kernelParams.m_fNoise.getSquaredValue() + m_kernelParams.m_sNoise.getSquaredValue();
}

#endif /* GAUSSIANPROCESS_KERNEL_H_ */
