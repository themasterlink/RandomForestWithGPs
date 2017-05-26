/*
 * RandomForestGaussianProcess.h
 *
 *  Created on: 29.06.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTGAUSSIANPROCESS_RANDOMFORESTGAUSSIANPROCESS_H_
#define RANDOMFORESTGAUSSIANPROCESS_RANDOMFORESTGAUSSIANPROCESS_H_

#include "../RandomForests/RandomForest.h"
#include "../Utility/ThreadSafeOutput.h"
#include <boost/thread.hpp> // Boost threads
#include <boost/bind.hpp> // Boost threads
#include "../GaussianProcess/GaussianProcess.h"
#include "../Data/LabeledVectorX.h"

#define GP_USED -1

class RFGPWriter;

class RandomForestGaussianProcess : public PredictorMultiClass {
	friend RFGPWriter;
public:

	RandomForestGaussianProcess(const DataSets& data, const int heightOfTrees = 0,
			const int amountOfTrees = 0, const std::string& folderPath = std::string(""));

	virtual ~RandomForestGaussianProcess();

	unsigned int predict(const VectorX& data, std::vector<real>& prob) const;

	unsigned int predict(const VectorX& point) const;

	void predictData(const Data& data, Labels& labels) const{
		UNUSED(data); UNUSED(labels);
		printError("This function is not implemented!");
	}

	void predictData(const Data& points, Labels& labels, std::vector< std::vector<real> >& probabilities) const{
		UNUSED(points); UNUSED(labels); UNUSED(probabilities);
		printError("Not implemented yet!");
	}

	unsigned int amountOfClasses() const {return m_amountOfUsedClasses;};

	void train();

private:

	void trainInParallel(const unsigned int iActClass, const int amountOfDataInRfRes,
			const int amountOfHyperPoints,
			const int iActRfClass, const LabeledData& dataOfActRf,
			const std::vector<int>& classCounts,
			GaussianProcess* actGp);

	const DataSets& m_data;

	const unsigned int m_amountOfUsedClasses;
	unsigned int m_amountOfDataPoints;
	RandomForest m_forest;
	std::vector<int> m_pureClassLabelForRfClass;
	//std::vector< std::vector<bool> > m_isGpInUse;
	std::vector<std::vector<GaussianProcess*> > m_gps;
	std::vector<std::string> m_classNames; // save name for a class id

	ThreadSafeOutput m_output;

	int m_maxPointsUsedInGpSingleTraining;
	const std::string& m_folderPath;
	bool m_didLoadTree;
	int m_nrOfRunningThreads;
};

#endif /* RANDOMFORESTGAUSSIANPROCESS_RANDOMFORESTGAUSSIANPROCESS_H_ */
