/*
 * IVM.h
 *
 *  Created on: 27.09.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_IVM_H_
#define GAUSSIANPROCESS_IVM_H_

#include "../Data/LabeledVectorX.h"
#include <boost/math/distributions/normal.hpp> // for normal_distribution
#include "Kernel/GaussianKernel.h"
#include "Kernel/RandomForestKernel.h"
#include "../Base/InformationPackage.h"
#include "../RandomNumberGenerator/RandomUniformNr.h"
#include "../Data/OnlineStorage.h"
#include "../CMAES/cmaes.h"
#include "../CMAES/boundary_transformation.h"

class IVM : public PredictorBinaryClass {
public:
	enum class KernelType {
		GAUSS = 0,
		RF = 1
	};

	IVM(OnlineStorage<LabeledVectorX*>& storage, const bool isPartOfMultiIvm = false);

	void init(const unsigned int numberOfInducingPoints, const Vector2i& labelsForClasses,
			const bool doEPUpdate, const bool calcDifferenceMatrixAlone = true);

	void setNumberOfInducingPoints(unsigned int nr);

	bool train(const bool doSampling, const int verboseLevel, const bool useKernelValuesAsBestParams = false);

	bool trainOptimizeStep(const int verboseLevel = 0);

	Real predict(const VectorX& input) const;

	Real predictMu(const VectorX& input) const;

	Real predictSigma(const VectorX& input) const;

	const List<unsigned int>& getSelectedInducingPoints() const { return m_I; };

	void setDerivAndLogZFlag(const bool doLogZ, const bool doDerivLogZ);

	unsigned int getLabelForOne() const;

	unsigned int getLabelForMinusOne() const;

	void setInformationPackage(InformationPackage* package){
		if(package != nullptr){
			m_package = package;
		}else{
			printError("The given information package was null!");
		}
	};

	void setKernelSeed(unsigned int seed);

	GaussianKernel* getGaussianKernel(){ return m_gaussKernel; }

	void setOnlineRandomForest(OnlineRandomForest* forest);

	bool isTrained(){ return m_trained; }

	KernelType getKernelType(){ return m_kernelType; }

	virtual ~IVM();

	Real m_logZ;
	GaussianKernelParams m_derivLogZ;

	void setClassName(const int classNrOfMulti = UNDEF_CLASS_LABEL){
		std::stringstream str2;
		if(classNrOfMulti != UNDEF_CLASS_LABEL){
			str2 << ClassKnowledge::getNameFor(classNrOfMulti) << "_";
		}
		str2 << ClassKnowledge::getNameFor(getLabelForOne());
		m_className = str2.str();
	}

	std::string getClassName(){ return m_className; };

private:

	Real predictOnTraining(const unsigned int id);

	Real calcInnerOfFindPointWhichDecreaseEntropyMost(const unsigned int j,
			const VectorX& zeta, const VectorX& mu,
			Real& g_kn, Real& nu_kn,
			const Real fraction, const Vector2i& amountOfPointsPerClassLeft, const int useThisLabel, const int verboseLevel);

	Real cumulativeLog(const Real x);

	Real cumulativeDerivLog(const Real x);

	void calcDerivatives(const VectorX& muL1);

	void calcLogZ();

	bool internalTrain(bool clearActiveSet = true, const int verboseLevel = 0);

	void testOnTrainingsData(int & amountOfOneChecks, int& amountOfOnesCorrect, int& amountOfMinusOneChecks,
			int& amountOfMinusOnesCorrect, Real& correctness, const Real probDiff,
			const bool onlyUseOnes, const bool wholeDataSet, const List<unsigned int>& testPoints);

	Real calcErrorOnTrainingsData(const bool wholeDataSet, const List<unsigned int>& testPoints, Real& oneCorrect, Real& minusOneCorrect);


	OnlineStorage<LabeledVectorX*>& m_storage;
	Matrix m_M;
	Matrix m_K;
	Matrix m_L;
	Matrix m_eye;
	VectorX m_y;
	VectorX m_nuTilde;
	VectorX m_tauTilde;
	VectorX m_muTildePlusBias;
	unsigned int m_dataPoints;
	unsigned int m_numberOfInducingPoints;
	Real m_bias;
	Real m_lambda;
	bool m_doEPUpdate;
	Real m_splitOfClassOneInData;
	Real m_desiredPoint;
	Real m_desiredMargin;
	bool m_calcLogZ;
	bool m_calcDerivLogZ;
	bool m_trained;
	List<unsigned int> m_J, m_I;
	Vector2i m_labelsForClasses;

	Eigen::LLT<Matrix> m_choleskyLLT;

	GaussianKernel* m_gaussKernel;
	RandomForestKernel* m_rfKernel;

	KernelType m_kernelType;

	boost::math::normal m_logisticNormal;

	RandomUniformNr m_uniformNr;

	bool m_useNeighbourComparison;

	InformationPackage* m_package;

	bool m_isPartOfMultiIvm;

	std::string m_className;

	cmaes::cmaes_t m_evo; /* an CMA-ES type struct or "object" */
	cmaes::cmaes_boundary_transformation_t m_cmaesBoundaries;
	// double because for cmaes compatible
	double *m_arFunvals, *m_hyperParamsValues;

	static Mutex m_listMutex;
};

#endif /* GAUSSIANPROCESS_IVM_H_ */
