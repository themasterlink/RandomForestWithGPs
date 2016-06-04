/*
 * RandomForest.cc
 *
 *  Created on: 01.06.2016
 *      Author: Max
 */

#include "RandomForest.h"
#include <thread>

RandomForest::RandomForest(const int maxDepth, const int amountOfTrees, const int amountOfClasses)
		:m_maxDepth(maxDepth),
			m_amountOfTrees(amountOfTrees),
			m_amountOfClasses(amountOfClasses),
			m_counterIncreaseValue(2),
			m_trees(amountOfTrees, DecisionTree(maxDepth, amountOfClasses)){
}

RandomForest::~RandomForest(){
}

void RandomForest::train(const Data& data, const Labels& labels, const int amountOfUsedDims,
		const Eigen::Vector2i minMaxUsedData){
	if(data.size() != labels.size()){
		printError("Label and data size are not equal!");
		return;
	}else if(data.size() < 2){
		printError("There must be at least two points!");
		return;
	}else if(data[0].rows() < 2){
		printError("There should be at least 2 dimensions in the data");
	}else if(amountOfUsedDims > data[0].rows()){
		printError("Amount of dims can't be bigger than the dimension size!");
		return;
	}

	StopWatch sw;
	const int nrOfParallel = std::thread::hardware_concurrency();
	boost::thread_group group;
	TreeCounter counter;
	m_counterIncreaseValue = std::max(2, m_amountOfTrees / nrOfParallel / 100);
	std::vector<RandomNumberGeneratorForDT> generators;
	for(int i = 0; i < nrOfParallel; ++i){
		const int seed = i;
		generators.push_back(
				RandomNumberGeneratorForDT(data[0].rows(), minMaxUsedData[0], minMaxUsedData[1],
						data.size(), seed));
		const int start = (i / (double) nrOfParallel) * m_amountOfTrees;
		const int end = ((i + 1) / (double) nrOfParallel) * m_amountOfTrees;
		group.add_thread(new boost::thread(boost::bind(&RandomForest::trainInParallel, this, data, labels, amountOfUsedDims, generators[i], start, end, &counter)));
	}
	while(true){
		usleep(0.2 * 1e3);
		std::cout << "\r                                                                  \r";
		const int c = counter.getCounter();
		std::cout << "Trees trained: " << c / (double) m_amountOfTrees * 100.0 << " %"
				<< ",\testimated rest time: "
				<< ((double) (m_amountOfTrees - c)) * (sw.elapsedSeconds() / (double) c) << " sec";
		flush(std::cout);
		if(counter.getCounter() >= m_amountOfTrees){
			break;
		}
	}
	group.join_all(); // wait until all are finished!
	std::cout << "\rFinish training in : " << sw.elapsedSeconds()
			<< " sec                                                                 " << std::endl;
}

void RandomForest::trainInParallel(const Data& data, const Labels& labels,
		const int amountOfUsedDims, RandomNumberGeneratorForDT& generator, const int start,
		const int end, TreeCounter* counter){
	for(int i = start; i < end; ++i){
		m_trees[i].train(data, labels, amountOfUsedDims, generator);
		if(i % m_counterIncreaseValue == 0 && counter != NULL){
			counter->addToCounter(m_counterIncreaseValue); // is a thread safe add
		}
	}
}

int RandomForest::predict(const DataElement& point) const{
	const int nrOfParallel = std::thread::hardware_concurrency();
	if(!(m_amountOfTrees < nrOfParallel || true)){
		// do not use -> has a bug!
		boost::thread_group group;
		std::vector<std::vector<int> > vectors(nrOfParallel,
				std::vector<int>(m_amountOfClasses, 0));
		for(int i = 0; i < nrOfParallel; ++i){
			const int start = (i / (double) nrOfParallel) * m_amountOfTrees;
			const int end = ((i + 1) / (double) nrOfParallel) * m_amountOfTrees;
			group.add_thread(new boost::thread(boost::bind(&RandomForest::predictInParallel, this, point, vectors[i], start, end)));
		}
		group.join_all(); // wait until all are finished!
		for(int i = 1; i < nrOfParallel; ++i){
			for(int j = 0; j < m_amountOfClasses; ++j){
				vectors[0][j] += vectors[i][j];
			}
		}
		return std::distance(vectors[0].cbegin(),
				std::max_element(vectors[0].cbegin(), vectors[0].cend()));
	}else{ // no parallel execution:
		std::vector<int> values(m_amountOfClasses, 0);
		for(std::vector<DecisionTree>::const_iterator it = m_trees.cbegin(); it != m_trees.cend();
				++it){
			++values[it->predict(point)];
		}
		//std::cout << "First: " << values[0] << ", second: " << values[1] << std::endl;
		return std::distance(values.cbegin(), std::max_element(values.cbegin(), values.cend()));
	}
}

void RandomForest::predictInParallel(const DataElement& point, std::vector<int>& values,
		const int start, const int end) const{
	for(int i = start; i < end; ++i){
		++values[m_trees[i].predict(point)];
	}
}
