/*
 * OnlineRandomForest.h
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_ONLINERANDOMFOREST_H_
#define RANDOMFORESTS_ONLINERANDOMFOREST_H_

#include "DynamicDecisionTree.h"
#include "../Data/Data.h"
#include "../Data/OnlineStorage.h"
#include <list>

class OnlineRandomForest : public Observer {
public:
	typedef std::list<DynamicDecisionTree> DecisionTreesContainer;
	typedef std::list<DynamicDecisionTree>::iterator DecisionTreeIterator;
	typedef std::list<DynamicDecisionTree>::const_iterator DecisionTreeConstIterator;

	OnlineRandomForest(OnlineStorage<ClassPoint*>& storage, const int maxDepth, const int amountOfTrees, const int amountOfUsedClasses);

	virtual ~OnlineRandomForest();

	void train();

	int predict(const DataPoint& point) const;

	void predictData(const Data& points, Labels& labels) const;

	void getLeafNrFor(std::vector<int>& leafNrs);

	int getNrOfTrees() const { return m_trees.size(); };

	void update(Subject* caller, unsigned int event);

	void update();

private:
	const int m_amountOfTrees;

	const int m_maxDepth;

	const int m_amountOfClasses;

	int m_amountOfPointsUntilRetrain;

	int m_counterForRetrain;

	int m_amountOfUsedDims;

	Eigen::Vector2d m_minMaxUsedDataFactor;

	OnlineStorage<ClassPoint*>& m_storage;

	DecisionTreesContainer m_trees;

	DecisionTreeIterator findWorstPerformingTree(double& correctAmount);

	Eigen::Vector2i getMinMaxData();

};


#endif /* RANDOMFORESTS_ONLINERANDOMFOREST_H_ */
