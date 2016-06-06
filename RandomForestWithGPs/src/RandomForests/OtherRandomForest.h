/*
 * OtherRandomForest.h
 *
 *  Created on: 01.06.2016
 *      Author: Max
 */

#ifndef OTHERRANDOMFORESTS_OTHERRANDOMFOREST_H_
#define OTHERRANDOMFORESTS_OTHERRANDOMFOREST_H_

#include "OtherDecisionTree.h"
#include <boost/thread.hpp> // Boost threads
#include "../RandomNumberGenerator/RandomNumberGeneratorForDT.h"
#include "DecisionTreeData.h"

class OtherTreeCounter{
private:
	boost::mutex mutex;
	int counter;
	public:
	void addToCounter(const int val){
		mutex.lock();
		counter += val;
		mutex.unlock();
	}

	int getCounter(){
		return counter;
	}
	OtherTreeCounter() : counter(0){};
};

class OtherRandomForest{
public:
	OtherRandomForest(const int maxDepth, const int amountOfTrees, const int amountOfUsedClasses);
	virtual ~OtherRandomForest();

	void addForest(const OtherRandomForest& forest);

	void train(const Data& data, const Labels& labels, const int amountOfUsedDims,
			const Eigen::Vector2i minMaxUsedData);

	void init(const int amountOfTrees);

	void generateTreeBasedOnData(const DecisionTreeData& data, const int element);

	int predict(const DataElement& point) const;

	void predictData(const Data& points, Labels& labels) const;

	int getNrOfTrees() const { return m_trees.size(); };

	const std::vector<OtherDecisionTree>& getTrees() const{ return m_trees; };

	std::vector<OtherDecisionTree>& getTrees(){ return m_trees; };

private:
	const int m_maxDepth;

	const int m_amountOfTrees;

	const int m_amountOfClasses;

	int m_counterIncreaseValue;

	std::vector<OtherDecisionTree> m_trees;

	void trainInParallel(const Data& data, const Labels& labels, const int amountOfUsedDims,
			RandomNumberGeneratorForDT& generator, const int start, const int end,
			OtherTreeCounter* counter);

	void predictDataInParallel(const Data& points, Labels* labels, const int start,
			const int end) const;

};

#endif /* OTHERRANDOMFORESTS_OTHERRANDOMFOREST_H_ */
