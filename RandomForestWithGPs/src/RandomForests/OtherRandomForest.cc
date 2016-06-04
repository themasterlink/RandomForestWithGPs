/*
 * OtherRandomForest.cc
 *
 *  Created on: 01.06.2016
 *      Author: Max
 */

#include "OtherRandomForest.h"
#include <thread>

OtherRandomForest::OtherRandomForest(const int maxDepth, const int amountOfTrees, const int amountOfClasses):
	m_maxDepth(maxDepth), m_amountOfTrees(amountOfTrees), m_amountOfClasses(amountOfClasses), m_counterIncreaseValue(2), m_trees(amountOfTrees, OtherDecisionTree(maxDepth, amountOfClasses)){
}

OtherRandomForest::~OtherRandomForest(){
}


void OtherRandomForest::train(const Data& data, const Labels& labels, const int amountOfUsedDims, const Eigen::Vector2i minMaxUsedData){
	if(data.size() != labels.size()){
		printError("Label and data size are not equal!"); return;
	}else if(data.size() < 2){
		printError("There must be at least two points!"); return;
	}else if(data[0].rows() < 2){
		printError("There should be at least 2 dimensions in the data");
	}else if(amountOfUsedDims > data[0].rows()){
		printError("Amount of dims can't be bigger than the dimension size!"); return;
	}

	StopWatch sw;
	const int nrOfParallel = std::thread::hardware_concurrency();
	boost::thread_group group;
	OtherTreeCounter counter;
	m_counterIncreaseValue = std::max(2, m_amountOfTrees / nrOfParallel / 100);
	std::vector<RandomNumberGeneratorForDT> generators;
	for(int i = 0; i < nrOfParallel; ++i){
		const int seed = i;
		generators.push_back(RandomNumberGeneratorForDT(data[0].rows(), minMaxUsedData[0], minMaxUsedData[1], data.size(), seed));
		const int start = (i/(double)nrOfParallel) * m_amountOfTrees;
		const int end =  ((i+1)/(double)nrOfParallel) * m_amountOfTrees;
		group.add_thread(new boost::thread(boost::bind(&OtherRandomForest::trainInParallel, this, data, labels, amountOfUsedDims, generators[i], start, end, &counter)));
	}
	while(true){
		usleep(0.2 * 1e6);
		const int c = counter.getCounter();
		if(c != 0){
			std::cout << "\r                                                                                                   \r";
			const double time = ((double)(m_amountOfTrees - c)) * (sw.elapsedSeconds() / (double) c) ;
			if(time < 60){
				std::cout << "Trees trained: " << c / (double) m_amountOfTrees * 100.0 << " %" << ",\testimated rest time: " << time << " sec";
			}else if(time < 3600){
				std::cout << "Trees trained: " << c / (double) m_amountOfTrees * 100.0 << " %" << ",\testimated rest time: " << (int) time / 60 << " min and " << fmod(time, 60) << " sec";
			}else{
				std::cout << "Trees trained: " << c / (double) m_amountOfTrees * 100.0 << " %" << ",\testimated rest time: " << (int) time / 3600 << " hours and "<< fmod((int) time / 60, 60) << " min and " << fmod(time, 60) << " sec";
			}
			flush(std::cout);
		}
		if(counter.getCounter() >= m_amountOfTrees){
			break;
		}
	}
	group.join_all(); // wait until all are finished!
	std::cout << "\rFinish training in : " << sw.elapsedSeconds() << " sec                                                                 " << std::endl;
}

void OtherRandomForest::trainInParallel(const Data& data, const Labels& labels, const int amountOfUsedDims, RandomNumberGeneratorForDT& generator, const int start, const int end, OtherTreeCounter* counter){
	for(int i = start; i < end; ++i){
		m_trees[i].train(data, labels, amountOfUsedDims, generator);
		if(i % m_counterIncreaseValue == 0 && counter != NULL){
			counter->addToCounter(m_counterIncreaseValue); // is a thread safe add
		}
	}
}

int OtherRandomForest::predict(const DataElement& point) const{
	std::vector<int> values(m_amountOfClasses,0);
	for(std::vector<OtherDecisionTree>::const_iterator it = m_trees.cbegin(); it != m_trees.cend(); ++it){
		++values[it->predict(point)];
	}
	//std::cout << "First: " << values[0] << ", second: " << values[1] << std::endl;
	return std::distance(values.cbegin(), std::max_element(values.cbegin(), values.cend()));
}

void OtherRandomForest::predictData(const Data& points, Labels& labels) const{
	labels.resize(points.size());
	const int nrOfParallel = std::thread::hardware_concurrency();
	boost::thread_group group;
	for(int i = 0; i < nrOfParallel; ++i){
		const int start = (i/(double)nrOfParallel) * points.size();
		const int end =  ((i+1)/(double)nrOfParallel) * points.size();
		group.add_thread(new boost::thread(boost::bind(&OtherRandomForest::predictDataInParallel, this, points, &labels, start, end)));
	}
	group.join_all(); // wait until all are finished!
}

void OtherRandomForest::predictDataInParallel(const Data& points, Labels* labels, const int start, const int end) const{
	for(int i = start; i < end; ++i){
		(*labels)[i] = predict(points[i]);
	}
}
