/*
 * RandomForest.cc
 *
 *  Created on: 01.06.2016
 *      Author: Max
 */

#include "RandomForest.h"
#include <boost/thread.hpp> // Boost threads

RandomForest::RandomForest(const int maxDepth, const int amountOfTrees, const int amountOfClasses):
	m_maxDepth(maxDepth), m_amountOfTrees(amountOfTrees), m_amountOfClasses(amountOfClasses), m_trees(amountOfTrees, DecisionTree(maxDepth, amountOfClasses)){
}

RandomForest::~RandomForest(){
}


void RandomForest::train(const Data& data, const Labels& labels, const int amountOfUsedDims, const Eigen::Vector2i minMaxUsedData){
	StopWatch sw;
	const int nrOfParallel = 8;
	boost::thread_group group;
	for(int i = 0; i < nrOfParallel; ++i){
		const int start = (i/(double)nrOfParallel) * m_amountOfTrees;
		const int end =  ((i+1)/(double)nrOfParallel) * m_amountOfTrees;
		std::cout << "start: " << start << std::endl;
		std::cout << "end: " << end << std::endl;
		group.add_thread(new boost::thread(boost::bind(&RandomForest::trainInParallel, this, data, labels, amountOfUsedDims, minMaxUsedData, start, end)));
	}
	group.join_all(); // wait until all are finished!
	std::cout << "Time needed: " << sw.elapsedSeconds() << std::endl;
}

void RandomForest::trainInParallel(const Data& data, const Labels& labels, const int amountOfUsedDims, const Eigen::Vector2i minMaxUsedData, const int start, const int end){
	for(int i = start; i < end; ++i){
		m_trees[i].train(data, labels, amountOfUsedDims, minMaxUsedData);
	}
}

int RandomForest::predict(const DataElement& point) const{
	std::vector<int> values(m_amountOfClasses,0);
	for(std::vector<DecisionTree>::const_iterator it = m_trees.cbegin(); it != m_trees.cend(); ++it){
		++values[it->predict(point)];
	}
	return std::distance(values.cbegin(), std::max_element(values.cbegin(), values.cend()));
}

