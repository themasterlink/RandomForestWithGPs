/*
 * OnlineRandomForest.cc
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#include "OnlineRandomForest.h"
#include "../Utility/Util.h"
#include "../Base/Settings.h"

OnlineRandomForest::OnlineRandomForest(OnlineStorage<ClassPoint*>& storage,
		const int maxDepth,
		const int amountOfTrees,
		const int amountOfUsedClasses):
		m_amountOfTrees(amountOfTrees),
		m_maxDepth(maxDepth),
		m_amountOfClasses(amountOfUsedClasses),
		m_amountOfPointsUntilRetrain(0),
		m_counterForRetrain(0),
		m_amountOfUsedDims(0),
		m_factorForUsedDims(0.),
		m_storage(storage),
		m_firstTrainingDone(false){
	storage.attach(this);
	Settings::getValue("OnlineRandomForest.factorAmountOfUsedDims", m_factorForUsedDims);
	Settings::getValue("OnlineRandomForest.amountOfPointsUntilRetrain", m_amountOfPointsUntilRetrain);
	double val;
	Settings::getValue("OnlineRandomForest.minUsedDataFactor", val);
	m_minMaxUsedDataFactor[0] = val;
	Settings::getValue("OnlineRandomForest.maxUsedDataFactor", val);
	m_minMaxUsedDataFactor[1] = val;
}

OnlineRandomForest::~OnlineRandomForest(){
}

void OnlineRandomForest::train(){
	if(m_storage.size() < 2){
		printError("There must be at least two points!");
		return;
	}else if(m_storage.dim() < 2){
		printError("There should be at least 2 dimensions in the data");
		return;
	}
	m_amountOfUsedDims = m_factorForUsedDims * m_storage.dim();
	std::cout << "amount of used dims: " << m_amountOfUsedDims << std::endl;
	if(m_amountOfUsedDims > m_storage.dim()){
		printError("Amount of dims can't be bigger than the dimension size!");
		return;
	}else if(m_amountOfUsedDims <= 0){
		printError("Amount of dims must be bigger than the zero!");
		return;
	}
	std::vector<int> values(m_amountOfClasses, 0);
	const int seed = 0;
	const Eigen::Vector2i minMax = getMinMaxData();
	const unsigned int amountOfThreads = 8;
	m_generators.resize(amountOfThreads);
	for(unsigned int i = 0; i < amountOfThreads; ++i){
		m_generators[i] = new RandomNumberGeneratorForDT(m_storage.dim(), minMax[0],
				minMax[1], m_storage.size(), seed);
		attach(m_generators[i]);
		m_generators[i]->update(this, OnlineStorage<ClassPoint*>::APPENDBLOCK); // init training with just one element is not useful
	}
	for(unsigned int i = 0; i < m_amountOfTrees; ++i){
		m_trees.push_back(DynamicDecisionTree(m_storage, m_maxDepth, m_amountOfClasses));
	}
	int counter = 1;
	int threadCounter = 1;
	boost::thread_group group;
	const int nrOfParallel = std::min((int) m_trees.size(), (int) boost::thread::hardware_concurrency());
	DecisionTreeIterator oldIt = m_trees.begin();
	TreeCounter treeCounter;
	InLinePercentageFiller::setActMax(m_amountOfTrees + 1);
	for(DecisionTreeIterator it = m_trees.begin(); it != m_trees.end(); ++it, ++counter){
		if(threadCounter == nrOfParallel){
			group.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::trainInParallel, this, oldIt, m_trees.end(), *m_generators[threadCounter - 1], &treeCounter)));
			break;
		}
		if(counter == (int) (m_trees.size() / (double) nrOfParallel * threadCounter)){
			group.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::trainInParallel, this, oldIt, it, *m_generators[threadCounter - 1], &treeCounter)));
			oldIt = it;
			++threadCounter;
		}
	}
	while(treeCounter.getCounter() < m_amountOfTrees){
		usleep(0.2 * 1e6);
		InLinePercentageFiller::setActValueAndPrintLine(treeCounter.getCounter());
	}
	group.join_all();
	m_firstTrainingDone = true;
}

void OnlineRandomForest::trainInParallel(const DecisionTreeIterator& start, const DecisionTreeIterator& end, RandomNumberGeneratorForDT& generator, TreeCounter* counter){
	for(DecisionTreeIterator it = start; it != end; ++it){
		it->train(m_amountOfUsedDims, generator);
		counter->addOneToCounter();
	}
}

void OnlineRandomForest::update(Subject* caller, unsigned int event){
	notify(event);
	switch(event){
		case OnlineStorage<ClassPoint*>::APPEND:{
			if(m_counterForRetrain >= m_amountOfPointsUntilRetrain){
				update();
				m_counterForRetrain = 0;
			}
			break;
		}
		case OnlineStorage<ClassPoint*>::APPENDBLOCK:{
			update();
			m_counterForRetrain = 0;
			break;
		}
		case OnlineStorage<ClassPoint*>::ERASE:{
			printError("This update type is not supported here!");
			break;
		}
		default: {
			printError("This update type is not supported here!");
			break;
		}
	}
}

int OnlineRandomForest::amountOfClasses() const{
	return m_amountOfClasses;
}

bool OnlineRandomForest::update(){
	if(!m_firstTrainingDone){
		train();
	}else{
		printLine();
		std::list<std::pair<DecisionTreeIterator, double> >* list = new std::list<std::pair<DecisionTreeIterator, double> >();
		printLine();
		sortTreesAfterPerformance(*list);
		printLine();
		std::cout << list->size() << std::endl;
		printLine();
		for(std::list<std::pair<DecisionTreeIterator, double> >::const_iterator it = list->begin(); it != list->end(); ++it){
			std::cout << it->second << std::endl;
		}
		if(list->begin()->second > 90.){
			printDebug("No update needed!");
			return false;
		}
		boost::thread_group group;
		const int nrOfParallel = boost::thread::hardware_concurrency();
		boost::mutex* mutex = new boost::mutex();
		for(unsigned int i = 0; i < nrOfParallel; ++i){
			group.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::updateInParallel, this, list, 200, mutex)));
		}
		group.join_all();
	}
	return true;
}

void OnlineRandomForest::sortTreesAfterPerformance(std::list<std::pair<DecisionTreeIterator, double> >& list){
	for(DecisionTreeIterator itTree = m_trees.begin(); itTree != m_trees.end(); ++itTree){
		int correct = 0;
		for(OnlineStorage<ClassPoint*>::ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
			ClassPoint& point = *(*it);
			if(point.getLabel() == itTree->predict(point)){
				++correct;
			}
		}
		const double correctVal = correct / (double) m_storage.size() * 100.;
		if(list.size() == 0){
			list.push_back(std::pair<DecisionTreeIterator, double>(itTree, correctVal));
		}else{
			for(std::list<std::pair<DecisionTreeIterator, double> >::iterator it = list.begin(); it != list.end(); ++it){
				if(it->second > correctVal){
					list.insert(it, std::pair<DecisionTreeIterator, double>(itTree, correctVal));
					break;
				}
			}
		}
	}
}

void OnlineRandomForest::updateInParallel(std::list<std::pair<DecisionTreeIterator, double> >* list, const int amountOfSteps, boost::mutex* mutex){
	mutex->lock();
	std::pair<DecisionTreeIterator, double> pair = *list->begin(); // copy of the first element
	list->pop_front(); // remove it
	mutex->unlock();

	for(unsigned int i = 0; i < amountOfSteps; ++i){
		pair.first->train(m_amountOfUsedDims, *m_generators[0]); // retrain worst tree
		int correct = 0;
		for(OnlineStorage<ClassPoint*>::ConstIterator itPoint = m_storage.begin(); itPoint != m_storage.end(); ++itPoint){
			ClassPoint& point = **itPoint;
			if(point.getLabel() == pair.first->predict(point)){
				++correct;
			}
		}

		// add to list again!
		mutex->lock();
		const double correctVal = correct / (double) m_storage.size() * 100.;
		if(list->size() == 0){
			list->push_back(std::pair<DecisionTreeIterator, double>(pair.first, correctVal));
		}else{
			for(std::list<std::pair<DecisionTreeIterator, double> >::iterator it = list->begin(); it != list->end(); ++it){
				std::cout << "it->second: " << it->second << ", correct: " << correctVal << std::endl;
				if(it->second > correctVal){
					list->insert(it, std::pair<DecisionTreeIterator, double>(pair.first, correctVal));
				}
			}
		}
		pair = *list->begin(); // copy of the first element
		list->pop_front(); // remove it
		mutex->unlock();
	}
}

OnlineRandomForest::DecisionTreeIterator OnlineRandomForest::findWorstPerformingTree(double& correctAmount){
	int minCorrect = m_storage.size();
	DecisionTreeIterator itWorst = m_trees.end();
	for(DecisionTreeIterator itTree = m_trees.begin(); itTree != m_trees.end(); ++itTree){
		int correct = 0;
		for(OnlineStorage<ClassPoint*>::ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
			ClassPoint& point = *(*it);
			if(point.getLabel() == itTree->predict(point)){
				++correct;
			}
		}
		if(minCorrect > correct){
			minCorrect = correct;
			itWorst = itTree;
		}
	}
	correctAmount = minCorrect / (double) m_storage.size() * 100.;
	std::cout << "Worst correct is: " << correctAmount << std::endl;
	return itWorst;
}

int OnlineRandomForest::predict(const DataPoint& point) const {
	if(m_firstTrainingDone){
		std::vector<int> values(m_amountOfClasses, 0);
		for(DecisionTreeConstIterator it = m_trees.cbegin(); it != m_trees.cend(); ++it){
			++values[it->predict(point)];
		}
		return std::distance(values.cbegin(), std::max_element(values.cbegin(), values.cend()));
	}
	return -1;
}

void OnlineRandomForest::predictData(const Data& points, Labels& labels) const{
	labels.resize(points.size());
	boost::thread_group group;
	const int nrOfParallel = boost::thread::hardware_concurrency();
	for(unsigned int i = 0; i < nrOfParallel; ++i){
		const int start = i / (double) nrOfParallel * points.size();
		const int end = (i + 1) / (double) nrOfParallel * points.size();
		std::cout << start << ", for: " << points.size() << std::endl;
		group.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::predictDataInParallel, this, points, &labels, start, end)));
	}
	group.join_all();
}

void OnlineRandomForest::predictData(const ClassData& points, Labels& labels) const{
	labels.resize(points.size());
	boost::thread_group group;
	const int nrOfParallel = boost::thread::hardware_concurrency();
	for(unsigned int i = 0; i < nrOfParallel; ++i){
		const int start = i / (double) nrOfParallel * points.size();
		const int end = (i + 1) / (double) nrOfParallel * points.size();
		group.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::predictClassDataInParallel, this, points, &labels, start, end)));
	}
	group.join_all();
}

void OnlineRandomForest::predictData(const Data& points, Labels& labels, std::vector< std::vector<double> >& probabilities) const{
	labels.resize(points.size());
	probabilities.resize(points.size());
	boost::thread_group group;
	const int nrOfParallel = boost::thread::hardware_concurrency();
	for(unsigned int i = 0; i < nrOfParallel; ++i){
		const int start = i / (double) nrOfParallel * points.size();
		const int end = (i + 1) / (double) nrOfParallel * points.size();
		group.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::predictDataProbInParallel, this, points, &labels, &probabilities, start, end)));
	}
	group.join_all();
}

void OnlineRandomForest::predictDataProbInParallel(const Data& points, Labels* labels, std::vector< std::vector<double> >* probabilities, const int start, const int end) const{
	if(m_firstTrainingDone){
		for(unsigned int i = start; i < end; ++i){
			(*probabilities)[i].resize(m_amountOfClasses);
			for(DecisionTreeConstIterator it = m_trees.cbegin(); it != m_trees.cend(); ++it){
				(*probabilities)[i][it->predict(*points[i])] += 1;
			}
			unsigned int iMax = 0;
			double max = 0.;
			const double fac = 1. / m_trees.size();
			for(unsigned int j = 0; j < m_amountOfClasses; ++j){
				(*probabilities)[i][j] *= fac;
				if(max < (*probabilities)[i][j]){
					max = (*probabilities)[i][j];
					iMax = j;
				}
			}
			(*labels)[i] = iMax;
		}
	}
}

void OnlineRandomForest::predictDataInParallel(const Data& points, Labels* labels, const int start, const int end) const{
	for(unsigned int i = start; i < end; ++i){
		(*labels)[i] = predict(*points[i]);
	}
}

void OnlineRandomForest::predictClassDataInParallel(const ClassData& points, Labels* labels, const int start, const int end) const{
	for(unsigned int i = start; i < end; ++i){
		(*labels)[i] = predict(*points[i]);
	}
}

void OnlineRandomForest::getLeafNrFor(std::vector<int>& leafNrs){
	leafNrs = std::vector<int>(m_amountOfClasses, 0);
	for(OnlineStorage<ClassPoint*>::ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
		leafNrs[predict(**it)] += 1;
	}
}

Eigen::Vector2i OnlineRandomForest::getMinMaxData(){
	Eigen::Vector2i minMax;
	minMax[0] = m_minMaxUsedDataFactor[0] * m_storage.size();
	minMax[1] = m_minMaxUsedDataFactor[1] * m_storage.size();
	return minMax;
}

OnlineStorage<ClassPoint*>& OnlineRandomForest::getStorageRef(){
	return m_storage;
}

ClassTypeSubject OnlineRandomForest::classType() const{
	return ClassTypeSubject::ONLINERANDOMFOREST;
}
