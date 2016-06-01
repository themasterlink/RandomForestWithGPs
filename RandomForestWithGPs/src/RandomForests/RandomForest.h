/*
 * RandomForest.h
 *
 *  Created on: 01.06.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_RANDOMFOREST_H_
#define RANDOMFORESTS_RANDOMFOREST_H_

#include "DecisionTree.h"

class RandomForest {
public:
	RandomForest(const int maxDepth, const int amountOfTrees, const int amountOfUsedClasses);
	virtual ~RandomForest();

	void train(const Data& data, const Labels& labels, const int amountOfUsedDims, const Eigen::Vector2i minMaxUsedData);

	int predict(const DataElement& point) const;


private:
	const int m_maxDepth;

	const int m_amountOfTrees;

	const int m_amountOfClasses;

	std::vector<DecisionTree> m_trees;

	void trainInParallel(const Data& data, const Labels& labels, const int amountOfUsedDims, const Eigen::Vector2i minMaxUsedData, const int start, const int end);


};

#endif /* RANDOMFORESTS_RANDOMFOREST_H_ */
