/*
 * KernelBase.h
 *
 *  Created on: 31.10.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_KERNEL_KERNELBASE_H_
#define GAUSSIANPROCESS_KERNEL_KERNELBASE_H_

#include "../../Data/LabeledVectorX.h"
#include "../../Utility/Util.h"
#include "KernelType.h"
#include "../../RandomNumberGenerator/RandomGaussianNr.h"
#include "../../Data/OnlineStorage.h"
#include "../../Utility/ReadWriterHelper.h"
#include "../../Base/Settings.h"
#include "../../Data/ClassKnowledge.h"

template<typename KernelType, unsigned int nrOfParams = KernelType::paramsAmount>
class KernelBase {
public:
	using OwnKernelElement = typename KernelType::OwnKernelElement;
	using OwnKernelInitParams = typename KernelType::OwnKernelInitParams;

	KernelBase(const OwnKernelInitParams& initParams, const bool sampleNewParams = true);
	virtual ~KernelBase() = 0;

	void init(const Matrix& dataMat, const bool calcDifferenceMatrix, const bool useSharedDifferenceMatrix);

	void init(const LabeledData& data, const bool calcDifferenceMatrix, const bool useSharedDifferenceMatrix);

	bool isInit() const { return m_init; };

	void calcCovariance(Matrix& cov) const;

	void calcCovarianceDerivative(Matrix& cov, const OwnKernelElement* type) const;

	void calcCovarianceDerivativeForInducingPoints(Matrix& cov, const std::list<int>& activeSet, const OwnKernelElement* type) const;

	double getDifferences(const int row, const int col) const { return (double) (*m_differences)(row, col); };

	void setHyperParamsWith(const KernelType& params);

	KernelType& getHyperParams();

	const KernelType& getHyperParams() const;

	void newRandHyperParams();

	virtual void setSeed(const int seed);

	void subGradient(const KernelType& gradient, const double factor);

	bool wasDifferenceCalced(){ return m_calcedDifferenceMatrix; };

	void setGaussianRandomVariables(const std::vector<real>& means, const std::vector<real> sds);

	void addToHyperParams(const KernelType& params, const double factor = 1.0);

	unsigned int getNrOfParams(){ return nrOfParams; };

	void calcDifferenceMatrix(const int start, const int end, Eigen::MatrixXf* usedMatrix);

	static void calcDifferenceMatrix(const int start, const int end, Eigen::MatrixXf& usedMatrix, const OnlineStorage<LabeledVectorX*>& storage, InformationPackage* package = nullptr);

	virtual double calcDiagElement(unsigned int row) const = 0;

	virtual double calcDerivativeDiagElement(unsigned int row, const OwnKernelElement* type) const = 0;

	virtual void calcKernelVector(const VectorX& vector, const Matrix& dataMat, VectorX& res) const = 0;

	virtual std::string prettyString() const = 0;

	void setDifferenceMatrix(Eigen::MatrixXf* differenceMatrix){
		m_differences = differenceMatrix;
		m_calcedDifferenceMatrix = true;
	}

	double getSeed(){ return m_seed; };

protected:

	virtual bool kernelCanHaveDifferenceMatrix() const = 0;

	virtual double kernelFunc(const int row, const int col) const = 0;

	virtual double kernelFuncVec(const VectorX& lhs, const VectorX& rhs) const = 0;

	virtual double kernelFuncDerivativeToParam(const int row, const int col, const OwnKernelElement* type, const int element = -1) const = 0;

	Eigen::MatrixXf* m_differences;

	Matrix* m_pDataMat;

	LabeledData* m_pData;

	bool m_init;

	bool m_calcedDifferenceMatrix;

	unsigned int m_dataPoints;

	KernelType m_kernelParams;

	RandomGaussianNr* m_randomGaussians[nrOfParams];

	double m_seed;

};

#define __INCLUDE_KERNELBASE
#include "KernelBase_i.h"
#undef __INCLUDE_KERNELBASE


#endif /* GAUSSIANPROCESS_KERNEL_KERNELBASE_H_ */
