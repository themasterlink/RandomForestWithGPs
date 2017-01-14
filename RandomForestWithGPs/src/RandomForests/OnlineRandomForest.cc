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
#include "../Data/DataWriterForVisu.h"

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
		m_desiredAmountOfTrees(0),
		m_useBigDynamicDecisionTrees(false),
		m_amountOfUsedLayer(0,0){
	storage.attach(this);
	Settings::getValue("OnlineRandomForest.factorAmountOfUsedDims", m_factorForUsedDims);
	Settings::getValue("OnlineRandomForest.amountOfPointsUntilRetrain", m_amountOfPointsUntilRetrain);
	double val;
	Settings::getValue("OnlineRandomForest.minUsedDataFactor", val);
	m_minMaxUsedDataFactor[0] = val;
	Settings::getValue("OnlineRandomForest.maxUsedDataFactor", val);
	m_minMaxUsedDataFactor[1] = val;
	Settings::getValue("OnlineRandomForest.ownSamplingTime", m_ownSamplingTime, m_ownSamplingTime);
	Settings::getValue("OnlineRandomForest.useBigDynmaicDecisionTrees", m_useBigDynamicDecisionTrees);
//	setDesiredAmountOfTrees(250);
}

OnlineRandomForest::~OnlineRandomForest(){
	for(DecisionTreeIterator it = m_trees.begin(); it != m_trees.end(); ++it){
		SAVE_DELETE(*it);
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
	if(m_amountOfUsedDims > (int) m_storage.dim()){
		printError("Amount of dims can't be bigger than the dimension size!");
		return;
	}else if(m_amountOfUsedDims <= 0){
		printError("Amount of dims must be bigger than zero!");
		return;
	}
	std::vector<int> values(m_amountOfClasses, 0);
//	const int seed = 0;
	bool useFixedValuesForMinMaxUsedSplits = Settings::getDirectBoolValue("MinMaxUsedSplits.useFixedValuesForMinMaxUsedSplits");
	Eigen::Vector2i minMaxUsedSplits;
	if(useFixedValuesForMinMaxUsedSplits ){
		int minVal = 0, maxVal = 0;
		Settings::getValue("MinMaxUsedSplits.minValue", minVal);
		Settings::getValue("MinMaxUsedSplits.maxValue", maxVal);
		minMaxUsedSplits << minVal, maxVal;
	}else{
		double minVal = 0, maxVal = 0;
		Settings::getValue("MinMaxUsedSplits.minValueFractionDependsOnDataSize", minVal);
		Settings::getValue("MinMaxUsedSplits.maxValueFractionDependsOnDataSize", maxVal);
		minMaxUsedSplits << (int) (minVal * m_storage.size()),  (int) (maxVal * m_storage.size());
	}
	const unsigned int amountOfThreads = boost::thread::hardware_concurrency();
	m_generators.resize(amountOfThreads);
	int stepSizeOverData = 0;
	Settings::getValue("OnlineRandomForest.stepSizeOverData", stepSizeOverData);
	for(unsigned int i = 0; i < amountOfThreads; ++i){
		m_generators[i] = new RandomNumberGeneratorForDT(m_storage.dim(), minMaxUsedSplits[0],
				minMaxUsedSplits[1], m_storage.size(), (i + 1) * 827537, stepSizeOverData);
		attach(m_generators[i]);
		m_generators[i]->update(this, OnlineStorage<ClassPoint*>::APPENDBLOCK); // init training with just one element is not useful
	}
	const unsigned int nrOfParallel = boost::thread::hardware_concurrency();
	const double trainingsTime = m_ownSamplingTime > 0 ? m_ownSamplingTime : CommandSettings::get_samplingAndTraining();
	std::vector<InformationPackage*> packages(nrOfParallel, nullptr);
	if(m_maxDepth > 5 && m_useBigDynamicDecisionTrees && Settings::getDirectBoolValue("OnlineRandomForest.determineBestLayerAmount")){
		boost::thread_group layerGroup;
		std::list<std::pair<unsigned int, unsigned int> > layerValues; // from 2 to
		const int start = std::max(2, (int)std::ceil(m_maxDepth / (double) 20)); // at least 2 layers, but one layer can not be bigger than 20
		for(unsigned int i = start; m_maxDepth / i > 3; ++i){
			for(unsigned int j = 1; j < std::min(4u,i); ++j){
				layerValues.push_back(std::pair<unsigned int, unsigned int>(i,j));
				printOnScreen("Try: " << i << ", " << j);
			}
		}
//		layerValues.push_back(4);
		const double secondsSpendPerSplit = 10;
		boost::mutex layerMutex;
		std::pair<int, int> bestLayerSplit(-1, -1);
		double bestCorrectness = 0;
		for(unsigned int i = 0; i < std::min(nrOfParallel, (unsigned int) layerValues.size()); ++i){
			layerGroup.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::tryAmountForLayers, this, m_generators[i], secondsSpendPerSplit, &layerValues, &layerMutex, &bestLayerSplit, &bestCorrectness)));
		}
		layerGroup.join_all();
		if(bestLayerSplit.first != -1){
			printOnScreen("Best amount of layers is: " << bestLayerSplit.first << ", " << bestLayerSplit.second << " with: " << bestCorrectness << " %%");
			m_amountOfUsedLayer = bestLayerSplit;
		}else{
			Settings::getValue("OnlineRandomForest.layerAmountOfBigDDT", m_amountOfUsedLayer.first);
			m_amountOfUsedLayer.second = 2; // default
		}
	}else if(m_useBigDynamicDecisionTrees && m_maxDepth > 5){
		Settings::getValue("OnlineRandomForest.layerAmountOfBigDDT", m_amountOfUsedLayer.first);
		m_amountOfUsedLayer.second = 2; // default
	}else{
		m_useBigDynamicDecisionTrees = false;
	}
	std::vector<std::vector<unsigned int> >* counterForClasses = nullptr;
	if(Settings::getDirectBoolValue("OnlineRandomForest.printErrorForTraining")){
		counterForClasses = new std::vector<std::vector<unsigned int> >(m_storage.size(), std::vector<unsigned int>(amountOfClasses(), 0));
	}
	boost::mutex mutexForCounter;
	boost::thread_group group;
	for(unsigned int i = 0; i < nrOfParallel; ++i){
		packages[i] = new InformationPackage(m_desiredAmountOfTrees == 0 ? InformationPackage::ORF_TRAIN : InformationPackage::ORF_TRAIN_FIX, 0, (m_trees.size() / (double) nrOfParallel));
		packages[i]->setStandartInformation("Train trees, thread nr: " + number2String(i));
		packages[i]->setTrainingsTime(trainingsTime);
		group.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::trainInParallel, this, m_generators[i], packages[i], m_desiredAmountOfTrees, counterForClasses, &mutexForCounter)));
	}
	int stillOneRunning = 1;
	if(m_desiredAmountOfTrees == 0){
		InLinePercentageFiller::setActMaxTime(trainingsTime);
	}else{
		InLinePercentageFiller::setActMax(m_desiredAmountOfTrees);
	}
//	double nextCheck = std::min(10.,m_ownSamplingTime / 10.);
	StopWatch sw;
	int lastCounter = 0;
	std::list<double> points;
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
		if(counterForClasses != nullptr && m_trees.size() > 0 && m_trees.size() - lastCounter >= 1){
			lastCounter = m_trees.size();
			int correct = 0;
			mutexForCounter.lock();
			for(unsigned int i = 0; i < m_storage.size(); ++i){
				if(m_storage[i]->getLabel() == std::distance((*counterForClasses)[i].cbegin(), std::max_element((*counterForClasses)[i].cbegin(), (*counterForClasses)[i].cend()))){
					++correct;
				}
			}
			mutexForCounter.unlock();
			points.push_back(correct / (double) m_storage.size() * 100.);
		}
		usleep(0.005 * 1e6);
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
		SAVE_DELETE(packages[i]);
	}
	if(counterForClasses != nullptr && points.size() > 0){
		DataWriterForVisu::writeSvg("correct.svg", points, true);
		openFileInViewer("correct.svg");
	}
	SAVE_DELETE(counterForClasses); // can be null is no problem
	m_firstTrainingDone = true;
}


void OnlineRandomForest::trainInParallel(RandomNumberGeneratorForDT* generator, InformationPackage* package, const unsigned int amountOfTrees,
		std::vector<std::vector<unsigned int> >* counterForClasses, boost::mutex* mutexForCounter){
	ThreadMaster::appendThreadToList(package);
	package->wait();
	int i = 0;
	const bool printErrorGraph = counterForClasses != nullptr;
	Labels* labels = nullptr;
	if(printErrorGraph){
		labels = new Labels(m_storage.size(), UNDEF_CLASS_LABEL);
	}
	while(true){ // the thread master will eventually kill this training
		m_treesMutex.lock();
		if(amountOfTrees > 0 && (unsigned int) m_trees.size() == amountOfTrees){
			m_treesMutex.unlock();
			break;
		}
		m_treesMutex.unlock();
		// performing this outside the lock makes the lock shorter (because the constructor calls contains a lot of memory allocation)
		DynamicDecisionTreeInterface* treePointer = nullptr;
		if(m_useBigDynamicDecisionTrees){
			treePointer = new BigDynamicDecisionTree(m_storage, m_maxDepth, m_amountOfClasses, m_amountOfUsedLayer.first, m_amountOfUsedLayer.second);
		}else{
			treePointer = new DynamicDecisionTree(m_storage, m_maxDepth, m_amountOfClasses);
		}
		m_treesMutex.lock();
		m_trees.push_back(treePointer);
		// create a new element and train it
		DynamicDecisionTreeInterface* tree = m_trees.back();
		m_treesMutex.unlock();
		tree->train(m_amountOfUsedDims, *generator);
		printInPackageOnScreen(package, "Number " << i++ << " was calculated");
		if(printErrorGraph){
			for(unsigned int i = 0; i < m_storage.size(); ++i){
				(*labels)[i] = tree->predict(*m_storage[i]);
			}
			//lock
			mutexForCounter->lock();
			for(unsigned int i = 0; i < m_storage.size(); ++i){
				(*counterForClasses)[i][(*labels)[i]] += 1;
			}
			mutexForCounter->unlock();
			//unlock
		}
		if(package->shouldTrainingBePaused()){
			package->wait();
		}else if(package->shouldTrainingBeAborted()){ // if amountOfTrees == 0 -> ORF_TRAIN_FIX -> can not be aborted
			break;
		}
	}
	printOnScreen("Task finished!");
	package->finishedTask();
}

void OnlineRandomForest::tryAmountForLayers(RandomNumberGeneratorForDT* generator, const double secondsPerSplit, std::list<std::pair<unsigned int, unsigned int> >* layerValues,
		boost::mutex* mutex, std::pair<int, int>* bestLayerSplit, double* bestCorrectness){
	while(true){
		mutex->lock();
		if(layerValues->size() > 0){
			const int layerAmount = layerValues->front().first;
			const int amountOfFastLayers = layerValues->front().second;
			layerValues->pop_front();
			mutex->unlock();
			StopWatch sw;
			DecisionTreesContainer trees;
			while(sw.elapsedSeconds() < secondsPerSplit){
//				printOnScreen("Train new tree! " << trees.size());
				trees.push_back(new BigDynamicDecisionTree(m_storage, m_maxDepth, m_amountOfClasses, layerAmount, amountOfFastLayers));
				trees.back()->train(m_amountOfUsedDims, *generator);
			}
//			unsigned int correctAmount = 0;
//			for(unsigned int i = 0; i < m_storage.size(); ++i){
//				std::vector<unsigned int> classes(amountOfClasses(), 0);
//				for(DecisionTreeConstIterator it = trees.begin(); it != trees.end(); ++it){
//					++classes[(**it).predict(*m_storage[i])];
//				}
//				if(m_storage[i]->getLabel() == std::distance(classes.cbegin(), std::max_element(classes.cbegin(), classes.cend()))){
//					++correctAmount;
//				}
//			}
			const double corr = trees.size(); //correctAmount / (double) m_storage.size() * 100. ;
			mutex->lock();
			printOnScreen("Test: " << layerAmount << ", " << amountOfFastLayers << ", with " << corr << " trees");
			if(corr > *bestCorrectness){
				bestLayerSplit->first = layerAmount;
				bestLayerSplit->second = amountOfFastLayers;
				*bestCorrectness = corr;
				printOnScreen("New best layer amount: " << layerAmount << ", " << amountOfFastLayers << ", with " << *bestCorrectness << " trees");
			}
			mutex->unlock();
			for(DecisionTreeIterator it = trees.begin(); it != trees.end(); ++it){
				SAVE_DELETE(*it);
			}
		}else{
			mutex->unlock();
			break;
		}
	}
}

void OnlineRandomForest::update(Subject* caller, unsigned int event){
	UNUSED(caller);
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
//		if(list->begin()->second > 90.){
//			printDebug("No update needed!");
//			return false;
//		}
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
			SAVE_DELETE(packages[i]);
		}
		SAVE_DELETE(mutex);
		SAVE_DELETE(list);
	}
	return true;
}

void OnlineRandomForest::sortTreesAfterPerformance(SortedDecisionTreeList& list){
	for(DecisionTreeIterator itTree = m_trees.begin(); itTree != m_trees.end(); ++itTree){
		int correct = 0;
		for(OnlineStorage<ClassPoint*>::ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
			const ClassPoint& point = *(*it);
			const DynamicDecisionTreeInterface* tree = *itTree;
			if(point.getLabel() == tree->predict(point)){
				++correct;
			}
		}
		const double correctVal = correct / (double) m_storage.size() * 100.;
		internalAppendToSortedList(&list, itTree, correctVal);
	}
}

void OnlineRandomForest::updateInParallel(SortedDecisionTreeList* list, const unsigned int amountOfSteps, boost::mutex* mutex, unsigned int threadNr, InformationPackage* package, int* counter){
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
	DynamicDecisionTreeInterface* switcher = nullptr;
	if(m_useBigDynamicDecisionTrees){
		switcher = new BigDynamicDecisionTree(m_storage, m_maxDepth, m_amountOfClasses, m_amountOfUsedLayer.first, m_amountOfUsedLayer.second);
	}else{
		switcher = new DynamicDecisionTree(m_storage, m_maxDepth, m_amountOfClasses);
	}
	for(unsigned int i = 0; i < amountOfSteps; ++i){
		switcher->train(m_amountOfUsedDims, *m_generators[threadNr]); // retrain worst tree
		int correct = 0;
		for(OnlineStorage<ClassPoint*>::ConstIterator itPoint = m_storage.begin(); itPoint != m_storage.end(); ++itPoint){
			ClassPoint& point = **itPoint;
			if(point.getLabel() == switcher->predict(point)){
				++correct;
			}
		}
		const double correctVal = correct / (double) m_storage.size() * 100.;
		if(correctVal > pair.second){
			DynamicDecisionTreeInterface* tree = *pair.first;
			// if the switcher performs better than the original tree, change both so
			// that the switcher (with the better result is placed in the list), and the original element gets the new switcher
			*pair.first = switcher;
			switcher = tree;
			// add to list again!
			printInPackageOnScreen(package, "Performed new step with better correctness of: " << number2String(correctVal, 2) << " %%");
		}else{
			// no switch -> the switcher is trys to improve itself
			printInPackageOnScreen(package, "Performed new step with worse correctness of " << number2String(correctVal, 2) << " %% not used");
		}
		mutex->lock();
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
	SAVE_DELETE(switcher);
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
		DynamicDecisionTreeInterface* tree = *itTree;
		for(OnlineStorage<ClassPoint*>::ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
			ClassPoint& point = *(*it);
			if(point.getLabel() == tree->predict(point)){
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

unsigned int OnlineRandomForest::predict(const DataPoint& point) const {
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

double OnlineRandomForest::predict(const DataPoint& point1, const DataPoint& point2, const unsigned int sampleAmount) const{
	if(m_firstTrainingDone){
			if(sampleAmount > (unsigned int) m_trees.size()){
				printError("This should never happen, redesign to produce more trees if needed");
			}
			int inSameClassCounter = 0;
			unsigned int counter = 0;
			for(DecisionTreeConstIterator it = m_trees.cbegin(); it != m_trees.cend() && counter < sampleAmount; ++it, ++counter){
				if((*it)->predict(point1) == (*it)->predict(point2)){ // labels are the same -> in the same cluster
					++inSameClassCounter;
				}
			}
			return inSameClassCounter / (double) std::min(sampleAmount, (unsigned int) m_trees.size());
	}
	return -1;
}

double OnlineRandomForest::predictPartitionEquality(const DataPoint& point1, const DataPoint& point2, RandomUniformNr& uniformNr, unsigned int amountOfSamples) const{
	if(m_firstTrainingDone && m_trees.size() > 0){
		if(amountOfSamples > (unsigned int) m_trees.size() || m_maxDepth <= 3 || amountOfSamples == 0){
			printError("This should not happen!");
			return -1;
		}
		int sameLeaveCounter = 0;
		unsigned int counter = 0;
		for(DecisionTreesContainer::const_iterator it = m_trees.begin(); it != m_trees.end() && counter < amountOfSamples; ++it){
			const unsigned int height = uniformNr();
			if((*it)->predictIfPointsShareSameLeaveWithHeight(point1, point2, height)){
				++sameLeaveCounter;
			}
			++counter;
		}
		return sameLeaveCounter / (double) std::min(amountOfSamples, (unsigned int) m_trees.size());
	}
	return -1;
}

void OnlineRandomForest::predictData(const Data& points, Labels& labels) const{
	labels.resize(points.size());
	boost::thread_group group;
	const unsigned int nrOfParallel = boost::thread::hardware_concurrency();
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
	const unsigned int nrOfParallel = boost::thread::hardware_concurrency();
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
	const unsigned int nrOfParallel = boost::thread::hardware_concurrency();
	for(unsigned int i = 0; i < nrOfParallel; ++i){
		const int start = i / (double) nrOfParallel * points.size();
		const int end = (i + 1) / (double) nrOfParallel * points.size();
		group.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::predictDataProbInParallel, this, points, &labels, &probabilities, start, end)));
	}
	group.join_all();
}

void OnlineRandomForest::predictData(const ClassData& points, Labels& labels, std::vector< std::vector<double> >& probabilities) const{
	labels.resize(points.size());
	probabilities.resize(points.size());
	boost::thread_group group;
	const unsigned int nrOfParallel = boost::thread::hardware_concurrency();
	for(unsigned int i = 0; i < nrOfParallel; ++i){
		const int start = i / (double) nrOfParallel * points.size();
		const int end = (i + 1) / (double) nrOfParallel * points.size();
		group.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::predictClassDataProbInParallel, this, points, &labels, &probabilities, start, end)));
	}
	group.join_all();
}

void OnlineRandomForest::predictDataProbInParallel(const Data& points, Labels* labels, std::vector< std::vector<double> >* probabilities, const unsigned int start, const unsigned int end) const{
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

void OnlineRandomForest::predictClassDataProbInParallel(const ClassData& points, Labels* labels, std::vector< std::vector<double> >* probabilities, const unsigned int start, const unsigned int end) const{
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

void OnlineRandomForest::predictDataInParallel(const Data& points, Labels* labels, const unsigned int start, const unsigned int end) const{
	for(unsigned int i = start; i < end; ++i){
		(*labels)[i] = predict(*points[i]);
	}
}

void OnlineRandomForest::predictClassDataInParallel(const ClassData& points, Labels* labels, const unsigned int start, const unsigned int end) const{
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
