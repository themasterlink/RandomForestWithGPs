/*
 * GaussianProcessMultiBinary.h
 *
 *  Created on: 21.07.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_GAUSSIANPROCESSMULTIBINARY_H_
#define GAUSSIANPROCESS_GAUSSIANPROCESSMULTIBINARY_H_

#ifdef BUILD_OLD_CODE

#include "../Data/LabeledVectorX.h"
#include "GaussianProcess.h"
#include "../Utility/ThreadSafeOutput.h"
#include "../Utility/ThreadSafeThreadCounter.h"
#include "BestHyperParams.h"

class GaussianProcessMultiBinary : public PredictorMultiClass {
public:
	GaussianProcessMultiBinary(int amountOfUsedClasses);
	virtual ~GaussianProcessMultiBinary();

	void train(const LabeledData& data, const Labels* guessedLabels = NULL);

	unsigned int predict(const VectorX& point, std::vector<Real>& prob) const;

	unsigned int predict(const VectorX& point) const;

	void predictData(const Data& data, Labels& labels) const{
		UNUSED(data); UNUSED(labels);
		printErrorAndQuit("This function is not implemented!");
	}

	void predictData(const Data& points, Labels& labels, std::vector< std::vector<Real> >& probabilities) const{
		UNUSED(points); UNUSED(labels); UNUSED(probabilities);
		printErrorAndQuit("Not implemented yet!");
	}

	unsigned int amountOfClasses() const;

private:

	void trainInParallel(const int iActClass,
			const int amountOfHyperPoints, const LabeledData& data,
			const std::vector<int>& classCounts, GaussianProcess* actGp);

	void optimizeHyperParams(const unsigned int iActClass,
			const int amountOfHyperPoints, const LabeledData& data,
			const std::vector<int>& classCounts, const std::vector<bool>& elementsUsedForValidation,
			const Matrix& testDataMat, const VectorX& testYGpInit, BestHyperParams* bestHyperParams);

	const int m_amountOfUsedClasses;
	int m_amountOfDataPoints;
	int m_amountOfDataPointsForUseAllTestsPoints;
	int m_maxPointsUsedInGpSingleTraining;
//	vectord m_lowerBound;
//	vectord m_upperBound;
	ThreadSafeThreadCounter m_threadCounter;
	ThreadSafeOutput m_output;
	std::vector<GaussianProcess* > m_gps;
};

#endif // BUILD_OLD_CODE

#endif /* GAUSSIANPROCESS_GAUSSIANPROCESSMULTIBINARY_H_ */
