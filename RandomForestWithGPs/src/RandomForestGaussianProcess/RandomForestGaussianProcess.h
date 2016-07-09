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
	RandomForestGaussianProcess(const Data& data, const Labels& labels,
			const int heightOfTrees, const int amountOfTrees, const int amountOfUsedClasses);
	virtual ~RandomForestGaussianProcess();

	int predict(const DataElement& data, std::vector<double>& prob) const;

	int amountOfClasses() const {return m_amountOfUsedClasses;};

private:
	const Data& m_data;
	const Labels& m_labels;

	const int m_heightOfTrees;
	const int m_amountOfTrees;
	const double m_amountOfUsedClasses;
	RandomForest m_forest;
	std::vector<bool> m_isPure;
	std::vector<std::vector<GaussianProcessBinary> > m_gps;
};

#endif /* RANDOMFORESTGAUSSIANPROCESS_RANDOMFORESTGAUSSIANPROCESS_H_ */
