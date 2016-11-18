/*
 * KernelBase.h
 *
 *  Created on: 31.10.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_KERNEL_KERNELBASE_H_
#define GAUSSIANPROCESS_KERNEL_KERNELBASE_H_

#include "../../Data/ClassData.h"
#include "../../Utility/Util.h"
#include "KernelType.h"
#include "../../Utility/ReadWriterHelper.h"
#include "../../Base/Settings.h"
#include "../../Data/ClassKnowledge.h"

template<typename KernelType, unsigned int nrOfParams = KernelType::paramsAmount>
class KernelBase {
public:
	typedef typename KernelType::OwnKernelElement OwnKernelElement;
	typedef typename KernelType::OwnKernelInitParams OwnKernelInitParams;

	KernelBase(const OwnKernelInitParams& initParams);
	virtual ~KernelBase() = 0;

	void init(const Eigen::MatrixXd& dataMat, const bool calcDifferenceMatrix = true);

	void init(const ClassData& data, const bool calcDifferenceMatrix = true);

	bool isInit() const { return m_init; };

	void calcCovariance(Eigen::MatrixXd& cov) const;

	void calcCovarianceDerivative(Eigen::MatrixXd& cov, const OwnKernelElement* type) const;

	void calcCovarianceDerivativeForInducingPoints(Eigen::MatrixXd& cov, const std::list<int>& activeSet, const OwnKernelElement* type) const;

	double getDifferences(const int row, const int col) const { return (double) m_differences(row, col); };

	void setHyperParamsWith(const KernelType& params);

	KernelType& getHyperParams();

	const KernelType& getHyperParams() const;

	void newRandHyperParams();

	void setSeed(const int seed);

	void setGaussianRandomVariables(const std::vector<double>& means, const std::vector<double> sds);

	void addToHyperParams(const KernelType& params, const double factor = 1.0);

	unsigned int getNrOfParams(){ return nrOfParams; };

	virtual double calcDiagElement(unsigned int row) const = 0;

	virtual double calcDerivativeDiagElement(unsigned int row, const OwnKernelElement* type) const = 0;

	virtual void calcKernelVector(const Eigen::VectorXd& vector, const Eigen::MatrixXd& dataMat, Eigen::VectorXd& res) const = 0;

	virtual std::string prettyString() const = 0;

protected:

	virtual double kernelFunc(const int row, const int col) const = 0;

	virtual double kernelFuncVec(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const = 0;

	virtual double kernelFuncDerivativeToParam(const int row, const int col, const OwnKernelElement* type, const int element = -1) const = 0;

	Eigen::MatrixXd m_differences;

	Eigen::MatrixXd* m_pDataMat;

	ClassData* m_pData;

	bool m_init;

	bool m_calcedDifferenceMatrix;

	int m_dataPoints;

	KernelType m_kernelParams;

	RandomGaussianNr* m_randomGaussians[nrOfParams];

};

#define __INCLUDE_KERNELBASE
#include "KernelBase_i.h"
#undef __INCLUDE_KERNELBASE


#endif /* GAUSSIANPROCESS_KERNEL_KERNELBASE_H_ */
