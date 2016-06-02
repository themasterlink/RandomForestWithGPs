/*
 * RandomForest.h
 *
 *  Created on: 01.06.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_RANDOMFOREST_H_
#define RANDOMFORESTS_RANDOMFOREST_H_

#include "DecisionTree.h"
#include <boost/thread.hpp> // Boost threads


class TreeCounter{
private:
	boost::mutex mutex;
    int counter;
public:
    void addToCounter(const int val){
    	mutex.lock(); counter += val; mutex.unlock();
    }

    int getCounter(){
    	return counter;
    }
    TreeCounter(): counter(0){};
};

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

	int m_counterIncreaseValue;

	std::vector<DecisionTree> m_trees;

	void trainInParallel(const Data& data, const Labels& labels, const int amountOfUsedDims, const Eigen::Vector2i minMaxUsedData, const int start, const int end, TreeCounter* counter);


};

#endif /* RANDOMFORESTS_RANDOMFOREST_H_ */
