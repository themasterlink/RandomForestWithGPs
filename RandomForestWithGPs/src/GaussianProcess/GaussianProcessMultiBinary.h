/*
 * GaussianProcessMultiBinary.h
 *
 *  Created on: 21.07.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_GAUSSIANPROCESSMULTIBINARY_H_
#define GAUSSIANPROCESS_GAUSSIANPROCESSMULTIBINARY_H_

#include "../Data/ClassData.h"
#include "GaussianProcess.h"
#include "../Utility/ThreadSafeOutput.h"
#include "../Utility/ThreadSafeThreadCounter.h"
#include <boost/thread.hpp> // Boost threads
#include <boost/bind.hpp> // Boost threads
#include "BestHyperParams.h"
#include "../GaussianProcess/BayesOptimizer.h"


class GaussianProcessMultiBinary : public PredictorMultiClass {
public:
	GaussianProcessMultiBinary(int amountOfUsedClasses);
	virtual ~GaussianProcessMultiBinary();

	void train(const ClassData& data, const Labels* guessedLabels = NULL);

	int predict(const DataPoint& point, std::vector<double>& prob) const;

	int predict(const DataPoint& point) const;

	void predictData(const Data& data, Labels& labels) const{
		printError("This function is not implemented!");
	}

	void predictData(const Data& points, Labels& labels, std::vector< std::vector<double> >& probabilities) const{
		printError("Not implemented yet!");
	}

	unsigned int amountOfClasses() const;

private:

	void trainInParallel(const int iActClass,
			const int amountOfHyperPoints, const ClassData& data,
			const std::vector<int>& classCounts, GaussianProcess* actGp);

	void optimizeHyperParams(const int iActClass,
			const int amountOfHyperPoints, const ClassData& data,
			const std::vector<int>& classCounts, const std::vector<bool>& elementsUsedForValidation,
			const Eigen::MatrixXd& testDataMat, const Eigen::VectorXd& testYGpInit, BestHyperParams* bestHyperParams);

	const int m_amountOfUsedClasses;
	int m_amountOfDataPoints;
	int m_amountOfDataPointsForUseAllTestsPoints;
	int m_maxPointsUsedInGpSingleTraining;
	vectord m_lowerBound;
	vectord m_upperBound;
	ThreadSafeThreadCounter m_threadCounter;
	ThreadSafeOutput m_output;
	std::vector<GaussianProcess* > m_gps;
};

#endif /* GAUSSIANPROCESS_GAUSSIANPROCESSMULTIBINARY_H_ */
