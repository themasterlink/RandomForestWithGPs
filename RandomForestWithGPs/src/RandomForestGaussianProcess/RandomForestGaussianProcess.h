/*
 * RandomForestGaussianProcess.h
 *
 *  Created on: 29.06.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTGAUSSIANPROCESS_RANDOMFORESTGAUSSIANPROCESS_H_
#define RANDOMFORESTGAUSSIANPROCESS_RANDOMFORESTGAUSSIANPROCESS_H_

#include "../GaussianProcess/GaussianProcessBinary.h"
#include "../RandomForests/RandomForest.h"

class RandomForestGaussianProcess{
public:
	RandomForestGaussianProcess(const DataSets& data, const int heightOfTrees, const int amountOfTrees);
	virtual ~RandomForestGaussianProcess();

	int predict(const DataElement& data, std::vector<double>& prob) const;

	int amountOfClasses() const {return m_amountOfUsedClasses;};

private:
	const DataSets& m_data;

	const int m_heightOfTrees;
	const int m_amountOfTrees;
	const double m_amountOfUsedClasses;
	int m_amountOfDataPoints;
	RandomForest m_forest;
	std::vector<int> m_pureClassLabelForRfClass;
	std::vector<std::vector<GaussianProcessBinary> > m_gps;
};

#endif /* RANDOMFORESTGAUSSIANPROCESS_RANDOMFORESTGAUSSIANPROCESS_H_ */
