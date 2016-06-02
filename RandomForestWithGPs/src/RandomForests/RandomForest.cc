/*
 * RandomForest.cc
 *
 *  Created on: 01.06.2016
 *      Author: Max
 */

#include "RandomForest.h"


RandomForest::RandomForest(const int maxDepth, const int amountOfTrees, const int amountOfClasses):
	m_maxDepth(maxDepth), m_amountOfTrees(amountOfTrees), m_amountOfClasses(amountOfClasses), m_trees(amountOfTrees, DecisionTree(maxDepth, amountOfClasses)){
}

RandomForest::~RandomForest(){
}


void RandomForest::train(const Data& data, const Labels& labels, const int amountOfUsedDims, const Eigen::Vector2i minMaxUsedData){
	StopWatch sw;
	const int nrOfParallel = 8;
	boost::thread_group group;
	TreeCounter counter;
	m_counterIncreaseValue = std::max(2, m_amountOfTrees / nrOfParallel / 100);
	for(int i = 0; i < nrOfParallel; ++i){
		const int start = (i/(double)nrOfParallel) * m_amountOfTrees;
		const int end =  ((i+1)/(double)nrOfParallel) * m_amountOfTrees;
		group.add_thread(new boost::thread(boost::bind(&RandomForest::trainInParallel, this, data, labels, amountOfUsedDims, minMaxUsedData, start, end, &counter)));
	}
	while(true){
		usleep(0.1 * 1e3);
		std::cout << "\r                                                                  \r";
		const int c = counter.getCounter();
		std::cout << "Trees trained: " << c / (double) m_amountOfTrees * 100.0 << " %" << ",\testimated rest time: " << ((double)(m_amountOfTrees - c)) * (sw.elapsedSeconds() / (double) c) << " sec";
		flush(std::cout);
		if(counter.getCounter() >= m_amountOfTrees){
			break;
		}
	}
	group.join_all(); // wait until all are finished!
	std::cout << "\rFinish training in : " << sw.elapsedSeconds() << " sec                                                                  " << std::endl;
}

void RandomForest::trainInParallel(const Data& data, const Labels& labels, const int amountOfUsedDims, const Eigen::Vector2i minMaxUsedData, const int start, const int end, TreeCounter* counter){
	for(int i = start; i < end; ++i){
		m_trees[i].train(data, labels, amountOfUsedDims, minMaxUsedData);
		if(i % m_counterIncreaseValue == 0 && counter != NULL){
			counter->addToCounter(m_counterIncreaseValue); // is a thread safe add
		}
	}
}

int RandomForest::predict(const DataElement& point) const{
	std::vector<int> values(m_amountOfClasses,0);
	for(std::vector<DecisionTree>::const_iterator it = m_trees.cbegin(); it != m_trees.cend(); ++it){
		++values[it->predict(point)];
	}
	return std::distance(values.cbegin(), std::max_element(values.cbegin(), values.cend()));
}

