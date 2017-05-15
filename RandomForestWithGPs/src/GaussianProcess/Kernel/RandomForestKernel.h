/*
 * RandomForestKernel.h
 *
 *  Created on: 02.12.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_KERNEL_RANDOMFORESTKERNEL_H_
#define GAUSSIANPROCESS_KERNEL_RANDOMFORESTKERNEL_H_

#include "KernelBase.h"
#include "KernelType.h"
#include "../../Base/Observer.h"
#include "../../Data/OnlineStorage.h"
#include "../../RandomForests/OnlineRandomForest.h"

class RandomForestKernel : public KernelBase<RandomForestKernelParams>, public Observer {
public:
	RandomForestKernel(OnlineStorage<ClassPoint*>& storage, const int maxDepth, const int samplingAmount, const int amountOfUsedClasses, const bool createOrf);

	RandomForestKernel(OnlineStorage<ClassPoint*>& storage, const OwnKernelInitParams& initParams, const bool createOrf);
	virtual ~RandomForestKernel();

	void init();

	void update(Subject* subject, unsigned int event) override;

	double calcDiagElement(unsigned int row) const override;

	double calcDerivativeDiagElement(unsigned int row, const OwnKernelElement* type) const override;

	void calcKernelVector(const Eigen::VectorXd& vector, const Eigen::MatrixXd& dataMat, Eigen::VectorXd& res) const override;

	std::string prettyString() const override;

	double kernelFunc(const int row, const int col) const override;

	double kernelFuncVec(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

	double kernelFuncDerivativeToParam(const int row, const int col, const OwnKernelElement* type, const int element = -1) const override;

	void setSeed(const int seed) override;

	void setOnlineRandomForest(OnlineRandomForest* forest){ m_rf = forest; }

private:

	bool kernelCanHaveDifferenceMatrix() const override {return false;};

	enum class KernelMode {
		LABEL = 0,
		PARTITION = 1
	};
	RandomUniformNr m_heightSampler;

	OnlineRandomForest* m_rf;

	KernelMode m_mode;
};

#endif /* GAUSSIANPROCESS_KERNEL_RANDOMFORESTKERNEL_H_ */
