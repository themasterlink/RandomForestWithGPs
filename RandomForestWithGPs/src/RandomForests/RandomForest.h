/*
 * OtherRandomForest.h
 *
 *  Created on: 01.06.2016
 *      Author: Max
 */

#ifndef OTHERRANDOMFORESTS_OTHERRANDOMFOREST_H_
#define OTHERRANDOMFORESTS_OTHERRANDOMFOREST_H_

#include <boost/thread.hpp> // Boost threads
#include "../RandomNumberGenerator/RandomNumberGeneratorForDT.h"
#include "DecisionTree.h"
#include "DecisionTreeData.h"

class TreeCounter{
public:
	TreeCounter() : counter(0){};

	void addToCounter(const int val){
		mutex.lock();
		counter += val;
		mutex.unlock();
	}

	int getCounter() const{
		return counter;
	}
private:
	boost::mutex mutex;
	int counter;
};

class RandomForest{
public:
	typedef std::vector<DecisionTree> DecisionTreesContainer;

	RandomForest(const int maxDepth, const int amountOfTrees, const int amountOfUsedClasses);
	virtual ~RandomForest();

	void addForest(const RandomForest& forest);

	void train(const Data& data, const Labels& labels, const int amountOfUsedDims,
			const Eigen::Vector2i minMaxUsedData);

	void init(const int amountOfTrees);

	void generateTreeBasedOnData(const DecisionTreeData& data, const int element);

	int predict(const DataElement& point) const;

	void predictData(const Data& points, Labels& labels) const;

	int getNrOfTrees() const { return m_trees.size(); };

	const DecisionTreesContainer& getTrees() const{ return m_trees; };

	DecisionTreesContainer& getTrees(){ return m_trees; };

	void getLeafNrFor(const Data& data, const Labels& labels, std::vector<int>& leafNrs);

private:
	const int m_amountOfTrees;

	const int m_amountOfClasses;

	int m_counterIncreaseValue;

	DecisionTreesContainer m_trees;

	void trainInParallel(const Data& data, const Labels& labels, const int amountOfUsedDims,
			RandomNumberGeneratorForDT& generator, const int start, const int end,
			TreeCounter* counter);

	void predictDataInParallel(const Data& points, Labels* labels, const int start,
			const int end) const;

};

#endif /* OTHERRANDOMFORESTS_OTHERRANDOMFOREST_H_ */
