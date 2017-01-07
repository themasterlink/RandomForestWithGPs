/*
 * OnlineRandomForest.cc
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#include "OnlineRandomForest.h"
#include "../Utility/Util.h"
#include "../Base/Settings.h"
#include "../Base/CommandSettings.h"

OnlineRandomForest::OnlineRandomForest(OnlineStorage<ClassPoint*>& storage,
		const int maxDepth,
		const int amountOfUsedClasses):
		m_maxDepth(maxDepth),
		m_amountOfClasses(amountOfUsedClasses),
		m_amountOfPointsUntilRetrain(0),
		m_counterForRetrain(0),
		m_amountOfUsedDims(0),
		m_factorForUsedDims(0.),
		m_storage(storage),
		m_firstTrainingDone(false),
		m_ownSamplingTime(-1),
		m_desiredAmountOfTrees(0){
	storage.attach(this);
	Settings::getValue("OnlineRandomForest.factorAmountOfUsedDims", m_factorForUsedDims);
	Settings::getValue("OnlineRandomForest.amountOfPointsUntilRetrain", m_amountOfPointsUntilRetrain);
	double val;
	Settings::getValue("OnlineRandomForest.minUsedDataFactor", val);
	m_minMaxUsedDataFactor[0] = val;
	Settings::getValue("OnlineRandomForest.maxUsedDataFactor", val);
	Settings::getValue("OnlineRandomForest.ownSamplingTime", m_ownSamplingTime, m_ownSamplingTime);
	m_minMaxUsedDataFactor[1] = val;
}

OnlineRandomForest::~OnlineRandomForest(){
	for(DecisionTreeIterator it = m_trees.begin(); it != m_trees.end(); ++it){
		delete *it;
	}
}

void OnlineRandomForest::train(){
	if(m_storage.size() < 2){
		printError("There must be at least two points!");
		return;
	}else if(m_storage.dim() < 2){
		printError("There should be at least 2 dimensions in the data");
		return;
	}
	m_amountOfUsedDims = std::max((int) (m_factorForUsedDims * m_storage.dim()), 2);
	printOnScreen("Amount of used dims: " << m_amountOfUsedDims);
	printOnScreen("Amount of used data: " << m_storage.size());
	if(m_amountOfUsedDims > m_storage.dim()){
		printError("Amount of dims can't be bigger than the dimension size!");
		return;
	}else if(m_amountOfUsedDims <= 0){
		printError("Amount of dims must be bigger than zero!");
		return;
	}
	std::vector<int> values(m_amountOfClasses, 0);
//	const int seed = 0;
	bool useFixedValuesForMinMaxUsedData = Settings::getDirectBoolValue("MinMaxUsedData.useFixedValuesForMinMaxUsedData");
	Eigen::Vector2i minMaxUsedData;
	if(useFixedValuesForMinMaxUsedData){
		int minVal = 0, maxVal = 0;
		Settings::getValue("MinMaxUsedData.minValue", minVal);
		Settings::getValue("MinMaxUsedData.maxValue", maxVal);
		minMaxUsedData << minVal, maxVal;
	}else{
		double minVal = 0, maxVal = 0;
		Settings::getValue("MinMaxUsedData.minValueFraction", minVal);
		Settings::getValue("MinMaxUsedData.maxValueFraction", maxVal);
		minMaxUsedData << (int) (minVal * m_storage.size()),  (int) (maxVal * m_storage.size());
	}
	const unsigned int amountOfThreads = boost::thread::hardware_concurrency();
	m_generators.resize(amountOfThreads);
	for(unsigned int i = 0; i < amountOfThreads; ++i){
		m_generators[i] = new RandomNumberGeneratorForDT(m_storage.dim(), minMaxUsedData[0],
				minMaxUsedData[1], m_storage.size(), (i + 1) * 82734879237);
		attach(m_generators[i]);
		m_generators[i]->update(this, OnlineStorage<ClassPoint*>::APPENDBLOCK); // init training with just one element is not useful
	}
	boost::thread_group group;
	const int nrOfParallel = boost::thread::hardware_concurrency();
	const double trainingsTime = m_ownSamplingTime > 0 ? m_ownSamplingTime : CommandSettings::get_samplingAndTraining();
	std::vector<InformationPackage*> packages(nrOfParallel, nullptr);
	for(unsigned int i = 0; i < nrOfParallel; ++i){
		packages[i] = new InformationPackage(m_desiredAmountOfTrees == 0 ? InformationPackage::ORF_TRAIN : InformationPackage::ORF_TRAIN_FIX, 0, (m_trees.size() / (double) nrOfParallel));
		packages[i]->setStandartInformation("Train trees, thread nr: " + number2String(i));
		packages[i]->setTrainingsTime(trainingsTime);
		group.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::trainInParallel, this, m_generators[i], packages[i], m_desiredAmountOfTrees)));
	}
	int stillOneRunning = 1;
	if(m_desiredAmountOfTrees == 0){
		InLinePercentageFiller::setActMaxTime(trainingsTime);
	}else{
		InLinePercentageFiller::setActMax(m_desiredAmountOfTrees);
	}
//	double nextCheck = std::min(10.,m_ownSamplingTime / 10.);
	StopWatch sw;
	while(stillOneRunning != 0){
		stillOneRunning = 0;
		for(unsigned int i = 0; i < packages.size(); ++i){
			if(!packages[i]->isTaskFinished()){
				++stillOneRunning;
			}
		}
		if(m_desiredAmountOfTrees == 0){
			InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(m_trees.size());
		}else{
			InLinePercentageFiller::setActValueAndPrintLine(m_trees.size());
		}
		usleep(0.2 * 1e6);
	}
	group.join_all();
	printOnScreen("Calculated " << m_trees.size() << " trees with depth: " << m_maxDepth);
	if(m_desiredAmountOfTrees == 0){
		InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(m_trees.size(), true);
	}else{
		InLinePercentageFiller::setActValueAndPrintLine(m_trees.size());
	}
	for(unsigned int i = 0; i < nrOfParallel; ++i){
		ThreadMaster::threadHasFinished(packages[i]);
		delete packages[i];
	}
	m_firstTrainingDone = true;
}

void OnlineRandomForest::trainInParallel(RandomNumberGeneratorForDT* generator, InformationPackage* package, const int amountOfTrees){
	ThreadMaster::appendThreadToList(package);
	package->wait();
	int i = 0;
	while(true){ // the thread master will eventually kill this training
		m_treesMutex.lock();
		if(amountOfTrees > 0 && m_trees.size() == amountOfTrees){
			m_treesMutex.unlock();
			break;
		}
		// create a new element and train it
		m_trees.push_back(new DynamicDecisionTree(m_storage, m_maxDepth, m_amountOfClasses));
		DynamicDecisionTree& tree = *m_trees.back();
		m_treesMutex.unlock();
		tree.train(m_amountOfUsedDims, *generator);
		printInPackageOnScreen(package, "Number " << i << " was calculated");
		if(package->shouldTrainingBePaused()){
			package->wait();
		}else if(package->shouldTrainingBeAborted()){ // if amountOfTrees == 0 -> ORF_TRAIN_FIX -> can not be aborted
			break;
		}
	}
	printOnScreen("Task finished!");
	package->finishedTask();
}

void OnlineRandomForest::update(Subject* caller, unsigned int event){
	updateMinMaxValues(event); // first update the min and max values
	notify(event); // this should alert the generators two adjust to the new min and max values
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

unsigned int OnlineRandomForest::amountOfClasses() const{
	return m_amountOfClasses;
}

bool OnlineRandomForest::update(){
	if(!m_firstTrainingDone){
		StopWatch sw;
		train();
		printOnScreen("Needed for training: " << sw.elapsedAsTimeFrame());
	}else{
		std::list<std::pair<DecisionTreeIterator, double> >* list = new std::list<std::pair<DecisionTreeIterator, double> >();
		sortTreesAfterPerformance(*list);
		if(list->begin()->second > 90.){
			printDebug("No update needed!");
			return false;
		}
		boost::thread_group group;
		const unsigned int nrOfParallel = std::min((int) boost::thread::hardware_concurrency(), (int) m_trees.size());
		boost::mutex* mutex = new boost::mutex();
		if(list->size() != m_trees.size()){
			printError("The sorting process failed, list size is: " << list->size() << ", should be: " << m_trees.size());
			return false;
		}
		int counter = 0;
		const int totalAmount = m_trees.size() / nrOfParallel * nrOfParallel;
		const int amountOfElements = totalAmount / nrOfParallel;
		InLinePercentageFiller::setActMax(totalAmount + 1);
		std::vector<InformationPackage*> packages(nrOfParallel, nullptr);
		for(unsigned int i = 0; i < nrOfParallel; ++i){
			packages[i] = new InformationPackage(InformationPackage::ORF_TRAIN, 0., amountOfElements);
			group.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::updateInParallel, this, list, amountOfElements, mutex, i, packages[i], &counter)));
		}
		int stillOneRunning = 1;
		while(counter < totalAmount && stillOneRunning != 0){
			stillOneRunning = 0;
			for(unsigned int i = 0; i < nrOfParallel; ++i){
				if(!packages[i]->isTaskFinished()){
					stillOneRunning += 1;
				}
			}
			InLinePercentageFiller::setActValueAndPrintLine(counter);
			usleep(0.1 * 1e6);
		}
		group.join_all();
		for(unsigned int i = 0; i < nrOfParallel; ++i){
			ThreadMaster::threadHasFinished(packages[i]);
			delete packages[i];
		}
		delete mutex;
		delete list;
	}
	return true;
}

void OnlineRandomForest::sortTreesAfterPerformance(SortedDecisionTreeList& list){
	for(DecisionTreeIterator itTree = m_trees.begin(); itTree != m_trees.end(); ++itTree){
		int correct = 0;
		for(OnlineStorage<ClassPoint*>::ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
			ClassPoint& point = *(*it);
			DynamicDecisionTree& tree = **itTree;
			if(point.getLabel() == tree.predict(point)){
				++correct;
			}
		}
		const double correctVal = correct / (double) m_storage.size() * 100.;
		internalAppendToSortedList(&list, itTree, correctVal);
	}
}

void OnlineRandomForest::updateInParallel(SortedDecisionTreeList* list, const int amountOfSteps, boost::mutex* mutex, unsigned int threadNr, InformationPackage* package, int* counter){
	if(package == nullptr){
		printError("This thread has no valid information package: " + number2String(threadNr));
		return;
	}
	package->setStandartInformation("Orf updating thread Nr: " + number2String(threadNr));
	ThreadMaster::appendThreadToList(package);
	package->wait();
	mutex->lock();
	SortedDecisionTreePair pair = *list->begin(); // copy of the first element
	list->pop_front(); // remove it
	mutex->unlock();
	for(unsigned int i = 0; i < amountOfSteps; ++i){
		DynamicDecisionTree& tree = **pair.first;
		tree.train(m_amountOfUsedDims, *m_generators[threadNr]); // retrain worst tree
		int correct = 0;
		for(OnlineStorage<ClassPoint*>::ConstIterator itPoint = m_storage.begin(); itPoint != m_storage.end(); ++itPoint){
			ClassPoint& point = **itPoint;
			if(point.getLabel() == tree.predict(point)){
				++correct;
			}
		}
		// add to list again!
		mutex->lock();
		const double correctVal = correct / (double) m_storage.size() * 100.;
		printInPackageOnScreen(package, "Performed new step with correctness: " + number2String(correctVal, 2));
		*counter += 1; // is already protected in mutex lock
		internalAppendToSortedList(list, pair.first, correctVal); // insert decision tree again in the list
		pair = *list->begin(); // get new element
		list->pop_front(); // remove it
		mutex->unlock();
		if(package->shouldTrainingBePaused()){
			package->wait();
		}else if(package->shouldTrainingBeAborted()){
			break;
		}
	}
	package->finishedTask();
}

void OnlineRandomForest::internalAppendToSortedList(SortedDecisionTreeList* list, DecisionTreeIterator& itTree, double correctVal){
	if(list->size() == 0){
		list->push_back(SortedDecisionTreePair(itTree, correctVal));
	}else{
		bool added = false;
		for(SortedDecisionTreeList::iterator it = list->begin(); it != list->end(); ++it){
			if(it->second > correctVal){
				list->insert(it, SortedDecisionTreePair(itTree, correctVal));
				added = true;
				break;
			}
		}
		if(!added){
			list->push_back(SortedDecisionTreePair(itTree, correctVal));
		}
	}
}

OnlineRandomForest::DecisionTreeIterator OnlineRandomForest::findWorstPerformingTree(double& correctAmount){
	int minCorrect = m_storage.size();
	DecisionTreeIterator itWorst = m_trees.end();
	for(DecisionTreeIterator itTree = m_trees.begin(); itTree != m_trees.end(); ++itTree){
		int correct = 0;
		DynamicDecisionTree& tree = **itTree;
		for(OnlineStorage<ClassPoint*>::ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
			ClassPoint& point = *(*it);
			if(point.getLabel() == tree.predict(point)){
				++correct;
			}
		}
		if(minCorrect > correct){
			minCorrect = correct;
			itWorst = itTree;
		}
	}
	correctAmount = minCorrect / (double) m_storage.size() * 100.;
	printOnScreen("Worst correct is: " << correctAmount);
	return itWorst;
}

int OnlineRandomForest::predict(const DataPoint& point) const {
	if(m_firstTrainingDone){
		std::vector<int> values(m_amountOfClasses, 0);
		int k = 0;
		for(DecisionTreeConstIterator it = m_trees.cbegin(); it != m_trees.cend(); ++it){
			const unsigned int value = (*it)->predict(point);
			++values[value];
			++k;
		}
		return std::distance(values.cbegin(), std::max_element(values.cbegin(), values.cend()));
	}
	return UNDEF_CLASS_LABEL;
}

double OnlineRandomForest::predict(const DataPoint& point1, const DataPoint& point2, const int sampleAmount) const{
	if(m_firstTrainingDone){
			if(sampleAmount > m_trees.size()){
				printError("This should never happen, redesign to produce more trees if needed");
			}
			int inSameClassCounter = 0;
			int counter = 0;
			for(DecisionTreeConstIterator it = m_trees.cbegin(); it != m_trees.cend() && counter < sampleAmount; ++it, ++counter){
				if((*it)->predict(point1) == (*it)->predict(point2)){ // labels are the same -> in the same cluster
					++inSameClassCounter;
				}
			}
			return inSameClassCounter / (double) std::min(sampleAmount, (int) m_trees.size());
	}
	return -1;
}

double OnlineRandomForest::predictPartitionEquality(const DataPoint& point1, const DataPoint& point2, RandomUniformNr& uniformNr, int amountOfSamples) const{
	if(m_firstTrainingDone && m_trees.size() > 0){
		if(amountOfSamples > m_trees.size() || m_maxDepth <= 3 || amountOfSamples == 0){
			printError("This should not happen!");
			return -1;
		}
		int sameLeaveCounter = 0;
		int counter = 0;
		for(DecisionTreesContainer::const_iterator it = m_trees.begin(); it != m_trees.end() && counter < amountOfSamples; ++it){
			const int height = uniformNr();
			if((*it)->predictIfPointsShareSameLeaveWithHeight(point1, point2, height)){
				++sameLeaveCounter;
			}
			++counter;
		}
		return sameLeaveCounter / (double) std::min(amountOfSamples, (int) m_trees.size());
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
				(*probabilities)[i][(*it)->predict(*points[i])] += 1;
			}
			unsigned int iMax = UNDEF_CLASS_LABEL;
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

const OnlineStorage<ClassPoint*>& OnlineRandomForest::getStorageRef() const{
	return m_storage;
}

ClassTypeSubject OnlineRandomForest::classType() const{
	return ClassTypeSubject::ONLINERANDOMFOREST;
}

void OnlineRandomForest::updateMinMaxValues(unsigned int event){
	if(m_storage.size() == 0){
		return;
	}
	const unsigned int dim = m_storage.dim();
	if(dim != m_minMaxValues.size()){
		m_minMaxValues.resize(m_storage.dim());
		for(unsigned int i = 0; i < dim; ++i){
			m_minMaxValues[i][0] = DBL_MAX;
			m_minMaxValues[i][1] = NEG_DBL_MAX;
		}
	}
	switch(event){
	case OnlineStorage<ClassPoint*>::APPEND:{
		ClassPoint& point = *m_storage.last();
		for(unsigned int k = 0; k < dim; ++k){
			bool change = false;
			if(point[k] < m_minMaxValues[k][0]){
				m_minMaxValues[k][0] = point[k];
				change = true;
			}
			if(point[k] > m_minMaxValues[k][1]){
				m_minMaxValues[k][1] = point[k];
				change = true;
			}
		}
		break;
	}
	case OnlineStorage<ClassPoint*>::APPENDBLOCK:{
		const unsigned int start = m_storage.getLastUpdateIndex();
		for(unsigned int t = start; t < m_storage.size(); ++t){
			ClassPoint& point = *m_storage[t];
			for(unsigned int k = 0; k < dim; ++k){
				if(point[k] < m_minMaxValues[k][0]){
					m_minMaxValues[k][0] = point[k];
				}
				if(point[k] > m_minMaxValues[k][1]){
					m_minMaxValues[k][1] = point[k];
				}
			}
		}
		break;
	}
	default:{
		printError("This event is not handled here!");
		break;
	}
	}
}
