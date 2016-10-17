/*
 * OnlineRandomForest.h
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_ONLINERANDOMFOREST_H_
#define RANDOMFORESTS_ONLINERANDOMFOREST_H_

#include "DecisionTree.h"
#include "../Data/Data.h"
#include <list>

class OnlineRandomForest {
public:
	typedef std::list<DecisionTree> DecisionTreesContainer;
	typedef std::list<DecisionTree>::iterator DecisionTreeIterator;
	typedef std::list<DecisionTree>::const_iterator DecisionTreeConstIterator;

	OnlineRandomForest(const int maxDepth, const int amountOfTrees, const int amountOfUsedClasses);

	virtual ~OnlineRandomForest();

	void train(const ClassData& data, const int amountOfUsedDims,
			const Eigen::Vector2i minMaxUsedData);

	int predict(const DataPoint& point) const;

	void predictData(const Data& points, Labels& labels) const;

	void getLeafNrFor(const ClassData& data, std::vector<int>& leafNrs);

	int getNrOfTrees() const { return m_trees.size(); };

	void update(const ClassData& data, const int amountOfUsedDims, const Eigen::Vector2i minMaxUsedData);

private:
	const int m_amountOfTrees;

	const int m_amountOfClasses;

	DecisionTreesContainer m_trees;

	DecisionTreeIterator findWorstPerformingTree(const ClassData& data, double& correctAmount);

};


#endif /* RANDOMFORESTS_ONLINERANDOMFOREST_H_ */
