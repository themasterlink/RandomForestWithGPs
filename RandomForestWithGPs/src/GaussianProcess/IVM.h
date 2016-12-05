/*
 * IVM.h
 *
 *  Created on: 27.09.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_IVM_H_
#define GAUSSIANPROCESS_IVM_H_

#include "../Data/ClassData.h"
#include <boost/math/distributions/normal.hpp> // for normal_distribution
#include <list>
#include "Kernel/GaussianKernel.h"
#include "Kernel/RandomForestKernel.h"
#include "../Base/ThreadMaster.h"
#include "../RandomNumberGenerator/RandomUniformNr.h"
#include "../Data/OnlineStorage.h"

class IVM {
public:
	enum KernelType {
		GAUSS = 0,
		RF = 1
	};

	typedef typename Eigen::VectorXd Vector;
	typedef typename Eigen::MatrixXd Matrix;
	template <typename T>
	using List = std::list<T>;
	template <typename M, typename N>
	using Pair = std::list<M, N>;

	IVM(OnlineStorage<ClassPoint*>& storage, const bool isPartOfMultiIvm = false);

	void init(const unsigned int numberOfInducingPoints, const Eigen::Vector2i& labelsForClasses,
			const bool doEPUpdate, const bool calcDifferenceMatrixAlone = true);

	void setNumberOfInducingPoints(unsigned int nr);

	bool train(double timeForTraining = 0., const int verboseLevel = 0);

	bool trainOptimizeStep(const int verboseLevel = 0);

	double predict(const Vector& input) const;

	double predictMu(const Vector& input) const;

	double predictSigma(const Vector& input) const;

	const List<int>& getSelectedInducingPoints(){ return m_I; };

	void setDerivAndLogZFlag(const bool doLogZ, const bool doDerivLogZ);

	unsigned int getLabelForOne() const;

	unsigned int getLabelForMinusOne() const;

	void setInformationPackage(InformationPackage* package){ m_package = package; };

	void setKernelSeed(unsigned int seed);

	GaussianKernel* getGaussianKernel(){ return m_gaussKernel; }

	void setOnlineRandomForest(OnlineRandomForest* forest);

	bool isTrained(){ return m_trained; }

	KernelType getKernelType(){ return m_kernelType; }

	virtual ~IVM();

	double m_logZ;
	GaussianKernelParams m_derivLogZ;

private:

	double calcInnerOfFindPointWhichDecreaseEntropyMost(const unsigned int j,
			const Vector& zeta, const Vector& mu,
			double& g_kn, double& nu_kn,
			const double fraction, const Eigen::Vector2i& amountOfPointsPerClassLeft, const int verboseLevel);

	double cumulativeLog(const double x);

	double cumulativeDerivLog(const double x);

	void calcDerivatives(const Vector& muL1);

	void calcLogZ();

	bool internalTrain(bool clearActiveSet = true, const int verboseLevel = 0);

	OnlineStorage<ClassPoint*>& m_storage;
	Matrix m_M;
	Matrix m_K;
	Matrix m_L;
	Matrix m_eye;
	Vector m_y;
	Vector m_nuTilde;
	Vector m_tauTilde;
	Vector m_muTildePlusBias;
	unsigned int m_dataPoints;
	unsigned int m_numberOfInducingPoints;
	double m_bias;
	double m_lambda;
	bool m_doEPUpdate;
	double m_desiredPoint;
	double m_desiredMargin;
	bool m_calcLogZ;
	bool m_calcDerivLogZ;
	bool m_trained;
	List<int> m_J, m_I;
	Eigen::Vector2i m_labelsForClasses;

	Eigen::LLT<Eigen::MatrixXd> m_choleskyLLT;

	GaussianKernel* m_gaussKernel;
	RandomForestKernel* m_rfKernel;

	KernelType m_kernelType;

	boost::math::normal m_logisticNormal;

	RandomUniformNr m_uniformNr;

	bool m_useNeighbourComparison;

	InformationPackage* m_package;

	bool m_isPartOfMultiIvm;
};

#endif /* GAUSSIANPROCESS_IVM_H_ */
