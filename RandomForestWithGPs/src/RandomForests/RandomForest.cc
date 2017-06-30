/*
 * OtherRandomForest.cc
 *
 *  Created on: 01.06.2016
 *      Author: Max
 */

#include "RandomForest.h"

RandomForest::RandomForest(const int maxDepth, const int amountOfTrees,
		const int amountOfClasses)
		:	m_amountOfTrees(amountOfTrees),
			m_amountOfClasses(amountOfClasses),
			m_counterIncreaseValue(2),
			m_trees(m_amountOfTrees, DecisionTree(maxDepth, amountOfClasses)){
}

RandomForest::~RandomForest(){
}

void RandomForest::init(const int amountOfTrees){
	*(const_cast<int*>(&m_amountOfTrees)) = amountOfTrees; // change of const value
	m_trees = DecisionTreesContainer(m_amountOfTrees, DecisionTree(0, 0));
}

void RandomForest::generateTreeBasedOnData(const DecisionTreeData& data, const int element){
	*(const_cast<int*>(&m_amountOfClasses)) = data.amountOfClasses;
	if((int) m_trees.size() > element){
		m_trees[element].initFromData(data);
	}else{
		printError("The element is bigger than the size: " << element);
	}
}

void RandomForest::train(const LabeledData& data, const int amountOfUsedDims,
		const Vector2i& minMaxUsedData){
	if(data.size() < 2){
		printError("There must be at least two points!");
		return;
	}else if(data[0]->rows() < 2){
		printError("There should be at least 2 dimensions in the data");
	}else if(amountOfUsedDims > data[0]->rows()){
		printError("Amount of dims can't be bigger than the dimension size!");
		return;
	}

	StopWatch sw;
	const auto nrOfParallel = ThreadMaster::instance().getAmountOfThreads();
	boost::thread_group group;
	TreeCounter counter;
	m_counterIncreaseValue = std::min(std::max(2, static_cast<int>(m_amountOfTrees / nrOfParallel / 100)), 100);
	std::vector<RandomNumberGeneratorForDT*> generators;
	RandomNumberGeneratorForDT::BaggingInformation baggingInformation; // gets all info in constructor
	for(int i = 0; i < nrOfParallel; ++i){
		const int seed = i;
		generators.push_back(new RandomNumberGeneratorForDT((int) data[0]->rows(), minMaxUsedData[0], minMaxUsedData[1],
															(int) data.size(), seed, baggingInformation, false));
		const int start = static_cast<const int>(i / static_cast<Real>(nrOfParallel) * m_amountOfTrees);
		const int end =   static_cast<const int>((i + 1) / static_cast<Real>(nrOfParallel) * m_amountOfTrees);
		group.add_thread(new boost::thread(boost::bind(&RandomForest::trainInParallel, this, data, amountOfUsedDims, generators[i], start, end, &counter)));
	}
	while(counter.getCounter() < m_amountOfTrees){
		// just the update for the amount of training left:
		sleepFor(0.2);
		const int c = counter.getCounter();
		if(c != 0){
			std::cout << "\r                                                                                                   \r";
			TimeFrame time = sw.elapsedAsTimeFrame();
			const Real fac = (Real) (m_amountOfTrees - c) / (Real) c;
			time *= fac;
			std::cout << "Trees trained: " << c / (Real) m_amountOfTrees * 100.0 << " %" << ",\testimated rest time: " << time;
			flush(std::cout);
		}
	}
	group.join_all(); // wait until all are finished!
	for(int i = 0; i < nrOfParallel; ++i){
		SAVE_DELETE(generators[i]);
	}
	std::cout << "\rFinish training in : " << sw.elapsedSeconds() << " sec                                                                 " << std::endl;
}

void RandomForest::trainInParallel(const LabeledData& data,
		const int amountOfUsedDims, RandomNumberGeneratorForDT* generator, const int start,
		const int end, TreeCounter* counter){
	int iCounter = 0;
	for(int i = start; i < end; ++i){
		m_trees[i].train(data, amountOfUsedDims, *generator);
		if(i % m_counterIncreaseValue == 0 && counter != NULL){
			counter->addToCounter(m_counterIncreaseValue); // is a thread safe add
			iCounter += m_counterIncreaseValue;
		}
	}
}

unsigned int RandomForest::predict(const VectorX& point) const{
	std::vector<unsigned int> values((unsigned long) m_amountOfClasses, 0);
	for(auto it = m_trees.cbegin(); it != m_trees.cend(); ++it){
		++values[it->predict(point)];
	}
	//std::cout << "First: " << values[0] << ", second: " << values[1] << std::endl;
	return (unsigned int) argMax(values.cbegin(), values.cend());
}

unsigned int RandomForest::predict(const LabeledVectorX& point) const{
	std::vector<unsigned int> values((unsigned long) m_amountOfClasses, 0);
	for(auto it = m_trees.cbegin(); it != m_trees.cend(); ++it){
		++values[it->predict(point)];
	}
	//std::cout << "First: " << values[0] << ", second: " << values[1] << std::endl;
	return (unsigned int) argMax(values.cbegin(), values.cend());
}

void RandomForest::predictData(const Data& points, Labels& labels) const{
	labels.resize(points.size());
	const auto nrOfParallel = ThreadMaster::instance().getAmountOfThreads();
	boost::thread_group group;
	for(auto i = (decltype(nrOfParallel))(0); i < nrOfParallel; ++i){
		const int start = (int) ((i / (Real) nrOfParallel) * points.size());
		const int end = (int) (((i + 1) / (Real) nrOfParallel) * points.size());
		group.add_thread(new boost::thread(boost::bind(&RandomForest::predictDataInParallel,
				this, points, &labels, start, end)));
	}
	group.join_all(); // wait until all are finished!
}

void RandomForest::predictData(const LabeledData& points, Labels& labels) const{
	labels.resize(points.size());
	const auto nrOfParallel = ThreadMaster::instance().getAmountOfThreads();
	boost::thread_group group;
	for(auto i = (decltype(nrOfParallel))(0); i < nrOfParallel; ++i){
		const int start = (int) ((i / (Real) nrOfParallel) * points.size());
		const int end = (int) (((i + 1) / (Real) nrOfParallel) * points.size());
		group.add_thread(new boost::thread(boost::bind(&RandomForest::predictDataInParallelClass,
				this, points, &labels, start, end)));
	}
	group.join_all(); // wait until all are finished!
}

void RandomForest::predictDataInParallelClass(const LabeledData& points, Labels* labels, const int start,
		const int end) const{
	for(int i = start; i < end; ++i){
		(*labels)[i] = predict(*points[i]);
	}
}

void RandomForest::predictDataInParallel(const Data& points, Labels* labels, const int start,
		const int end) const{
	for(int i = start; i < end; ++i){
		(*labels)[i] = predict(*points[i]);
	}
}

void RandomForest::getLeafNrFor(const LabeledData& data, std::vector<int>& leafNrs){
	leafNrs = std::vector<int>(m_amountOfClasses, 0);
	for(unsigned int i = 0; i < (unsigned int) data.size(); ++i){
		leafNrs[predict(*data[i])] += 1;
	}
}


void RandomForest::addForest(const RandomForest& forest){
	if(forest.getNrOfTrees() == 0){
		printError("Can't add a empty forest!");
		return;
	}
	m_trees.reserve(forest.getTrees().size() + m_trees.size());
	for(auto it = forest.getTrees().cbegin(); it != forest.getTrees().cend(); ++it){
		m_trees.push_back(*it);
	}
}

unsigned int RandomForest::amountOfClasses() const{
	return m_amountOfClasses;
}

