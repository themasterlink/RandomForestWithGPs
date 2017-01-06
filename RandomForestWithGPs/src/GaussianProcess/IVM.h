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
#include "../Base/InformationPackage.h"
#include "../RandomNumberGenerator/RandomUniformNr.h"
#include "../Data/OnlineStorage.h"
#include "../CMAES/cmaes.h"
#include "../CMAES/boundary_transformation.h"

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

	IVM(OnlineStorage<ClassPoint*>& storage, const bool isPartOfMultiIvm = false);

	void init(const unsigned int numberOfInducingPoints, const Eigen::Vector2i& labelsForClasses,
			const bool doEPUpdate, const bool calcDifferenceMatrixAlone = true);

	void setNumberOfInducingPoints(unsigned int nr);

	bool train(const bool doSampling, const int verboseLevel, const bool useKernelValuesAsBestParams = false);

	bool trainOptimizeStep(const int verboseLevel = 0);

	double predict(const Vector& input) const;

	double predictMu(const Vector& input) const;

	double predictSigma(const Vector& input) const;

	const List<unsigned int>& getSelectedInducingPoints(){ return m_I; };

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

	double m_logZ;
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

	double predictOnTraining(const unsigned int id);

	double calcInnerOfFindPointWhichDecreaseEntropyMost(const unsigned int j,
			const Vector& zeta, const Vector& mu,
			double& g_kn, double& nu_kn,
			const double fraction, const Eigen::Vector2i& amountOfPointsPerClassLeft, const int useThisLabel, const int verboseLevel);

	double cumulativeLog(const double x);

	double cumulativeDerivLog(const double x);

	void calcDerivatives(const Vector& muL1);

	void calcLogZ();

	bool internalTrain(bool clearActiveSet = true, const int verboseLevel = 0);

	void testOnTrainingsData(int & amountOfOneChecks, int& amountOfOnesCorrect, int& amountOfMinusOneChecks,
			int& amountOfMinusOnesCorrect, double& correctness, const double probDiff,
			const bool onlyUseOnes, const bool wholeDataSet, const List<unsigned int>& testPoints);

	double calcErrorOnTrainingsData(const bool wholeDataSet, const List<unsigned int>& testPoints, double& oneCorrect, double& minusOneCorrect);


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
	double m_splitOfClassOneInData;
	double m_desiredPoint;
	double m_desiredMargin;
	bool m_calcLogZ;
	bool m_calcDerivLogZ;
	bool m_trained;
	List<unsigned int> m_J, m_I;
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

	std::string m_className;

	cmaes::cmaes_t m_evo; /* an CMA-ES type struct or "object" */
	cmaes::cmaes_boundary_transformation_t m_cmaesBoundaries;
	double *m_arFunvals, *m_hyperParamsValues;

	static boost::mutex m_listMutex;
};

#endif /* GAUSSIANPROCESS_IVM_H_ */
