/*
 * OnlineRandomForest.cc
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#include "OnlineRandomForest.h"
#include "../Base/Settings.h"
#include "../Data/DataWriterForVisu.h"

OnlineRandomForest::OnlineRandomForest(OnlineStorage<LabeledVectorX *> &storage,
									   const unsigned int maxDepth,
									   const int amountOfUsedClasses):
		m_maxDepth(maxDepth),
		m_amountOfClasses((unsigned int) amountOfUsedClasses),
		m_amountOfPointsUntilRetrain(0),
		m_counterForRetrain(0),
		m_amountOfUsedDims(0),
		m_factorForUsedDims((Real) 0.),
		m_storage(storage),
		m_firstTrainingDone(false),
		m_useBigDynamicDecisionTrees(false),
		m_amountOfUsedLayer(0,0),
		m_folderForSavedTrees("./"),
		m_savedAnyTreesToDisk(false),
		m_amountOfTrainedTrees(0),
		m_usedMemory(0){
	storage.attach(this);
	Settings::getValue("OnlineRandomForest.factorAmountOfUsedDims", m_factorForUsedDims);
	Settings::getValue("OnlineRandomForest.amountOfPointsUntilRetrain", m_amountOfPointsUntilRetrain);
	Settings::getValue("OnlineRandomForest.useBigDynamicDecisionTrees", m_useBigDynamicDecisionTrees);
	Settings::getValue("OnlineRandomForest.amountOfPointsCheckedPerSplit", m_amountOfPointsCheckedPerSplit);
	readTrainingsModeFromSetting();
}

OnlineRandomForest::~OnlineRandomForest(){
	for(DecisionTreeIterator it = m_trees.begin(); it != m_trees.end(); ++it){
		SAVE_DELETE(*it);
	}
}

void OnlineRandomForest::trainInParallel(RandomNumberGeneratorForDT* generator, InformationPackage* package, const unsigned int amountOfTrees,
		std::vector<std::vector<unsigned int> >* counterForClasses, boost::mutex* mutexForCounter){
	ThreadMaster::appendThreadToList(package);
	package->wait();
	printInPackageOnScreen(package, "Started training!");
	int counter = 0;
	const bool printErrorGraph = counterForClasses != nullptr;
	Labels* labels = nullptr;
	if(printErrorGraph){
		labels = new Labels(m_storage.size(), UNDEF_CLASS_LABEL);
	}
	while(true){ // the thread master will eventually kill this training
		m_treesMutex.lock();
		if(amountOfTrees > 0 && (unsigned int) m_trees.size() >= amountOfTrees){
			printOnScreen("Abort because amount of desired trees reached!");
			m_treesMutex.unlock();
			break;
		}
//		const unsigned int treeAmount = m_trees.size();
		m_treesMutex.unlock();
		// check if the memory consumption is to high -> write trees to disk
		mutexForCounter->lock();
//		printInPackageOnScreen(package, "Mem: " << getPercentageForUsedMemory());
		if(m_trainingsConfig.hasMemoryConstraint() && package->canBeAbortedInGeneral() &&  m_usedMemory >= m_trainingsConfig.m_memory){
			package->abortThread();
			printOnScreen("Abort because of memory, max: " << StringHelper::convertMemorySpace(m_trainingsConfig.m_memory) << ", has: " << StringHelper::convertMemorySpace(m_usedMemory));
		}
		mutexForCounter->unlock();
		// performing this outside the lock makes the lock shorter (because the constructor calls contains a lot of memory allocation)
		DynamicDecisionTreeInterface* treePointer = nullptr;
		if(m_useBigDynamicDecisionTrees){
			treePointer = new BigDynamicDecisionTree(m_storage, m_maxDepth, m_amountOfClasses, m_amountOfUsedLayer.first, m_amountOfUsedLayer.second, m_amountOfPointsCheckedPerSplit);
		}else{
			treePointer = new DynamicDecisionTree<unsigned int>(m_storage, m_maxDepth, m_amountOfClasses, m_amountOfPointsCheckedPerSplit);
		}
		treePointer->train((unsigned int) m_amountOfUsedDims, *generator);
		const MemoryType memForTree = treePointer->getMemSize();
		printInPackageOnScreen(package, "Number " << counter++ << " was calculated, total memory usage: " << StringHelper::convertMemorySpace(m_usedMemory + memForTree));
		if(printErrorGraph){
			for(unsigned int i = 0; i < m_storage.size(); ++i){
				(*labels)[i] = treePointer->predict(*m_storage[i]);
			}
			//lock
			mutexForCounter->lock();
			for(unsigned int i = 0; i < m_storage.size(); ++i){
				(*counterForClasses)[i][(*labels)[i]] += 1;
			}
			mutexForCounter->unlock();
			//unlock
		}
		m_treesMutex.lock();
		m_usedMemory += memForTree;
		m_trees.push_back(treePointer); // add it to list
		++m_amountOfTrainedTrees;
		m_treesMutex.unlock();
		if(package->shouldThreadBePaused()){
			package->wait();
		}else if(package->shouldThreadBeAborted()){ // if amountOfTrees != 0 -> ORF_TRAIN_FIX -> can not be aborted
			break;
		}
	}
	printOnScreen("Task finished!");
	package->finishedTask();
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
	Vector2i minMaxUsedSplits;
	if(useFixedValuesForMinMaxUsedSplits ){
		int minVal = 0, maxVal = 0;
		Settings::getValue("MinMaxUsedSplits.minValue", minVal);
		Settings::getValue("MinMaxUsedSplits.maxValue", maxVal);
		minMaxUsedSplits << minVal, maxVal;
	}else{
		Real minVal = (Real) 0, maxVal = (Real) 0;
		Settings::getValue("MinMaxUsedSplits.minValueFractionDependsOnDataSize", minVal);
		Settings::getValue("MinMaxUsedSplits.maxValueFractionDependsOnDataSize", maxVal);
		minMaxUsedSplits << (int) (minVal * m_storage.size()),  (int) (maxVal * m_storage.size());
	}
	const unsigned int amountOfThreads = ThreadMaster::getAmountOfThreads();
	m_generators.resize(amountOfThreads);
	int stepSizeOverData = 0;
	Settings::getValue("OnlineRandomForest.Tree.stepSizeOverData", stepSizeOverData);
	for(unsigned int i = 0; i < amountOfThreads; ++i){
		m_generators[i] = new RandomNumberGeneratorForDT(m_storage.dim(), minMaxUsedSplits[0],
														 minMaxUsedSplits[1], m_storage.size(), (i + 1) * 827537, stepSizeOverData);
		attach(m_generators[i]);
		m_generators[i]->update(this, OnlineStorage<LabeledVectorX*>::Event::APPENDBLOCK); // init training with just one element is not useful
	}
	const unsigned int usedAmountOfPackages = std::min(amountOfThreads, m_trainingsConfig.isTreeAmountMode() ? m_trainingsConfig.m_amountOfTrees : amountOfThreads);
	std::vector<InformationPackage*> packages(usedAmountOfPackages, nullptr);
	if(m_maxDepth > 7 && m_useBigDynamicDecisionTrees && Settings::getDirectBoolValue("OnlineRandomForest.determineBestLayerAmount")){
		boost::thread_group layerGroup;
		std::list<std::pair<unsigned int, unsigned int> > layerValues; // from 2 to
		const auto start = (unsigned int) std::max(2, (int)std::ceil(m_maxDepth / (Real) 12)); // at least 2 layers, but one layer can not be bigger than 20
		for(unsigned int i = start; m_maxDepth / i > 3; ++i){
			for(unsigned int j = 2; j < std::min(4u,i + 1); ++j){
				layerValues.push_back(std::pair<unsigned int, unsigned int>(i,j));
				printOnScreen("Try: " << i << ", " << j);
			}
		}
//		layerValues.push_back(4);
		const Real secondsSpendPerSplit = 180;
		boost::mutex layerMutex;
		std::pair<int, int> bestLayerSplit(-1, -1);
		Real bestCorrectness = 0;
		for(unsigned int i = 0; i < std::min(amountOfThreads, (unsigned int) layerValues.size()); ++i){
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
		Settings::getValue("OnlineRandomForest.layerFastAmountOfBigDDT", m_amountOfUsedLayer.second);
	}else{
		m_useBigDynamicDecisionTrees = false;
	}
	if(m_useBigDynamicDecisionTrees){
		printOnScreen("First layer amount: " << m_amountOfUsedLayer.first << ", amount of second layers: " << m_amountOfUsedLayer.second);
	}
	std::vector<std::vector<unsigned int> >* counterForClasses = nullptr;
	if(Settings::getDirectBoolValue("OnlineRandomForest.printErrorForTraining")){
		counterForClasses = new std::vector<std::vector<unsigned int> >(m_storage.size(), std::vector<unsigned int>(amountOfClasses(), 0));
	}
	const Real trainingsTimeForPackages = m_trainingsConfig.isTimeMode() ? m_trainingsConfig.m_seconds : 0;
	boost::mutex mutexForCounter;
	boost::thread_group group;
	for(unsigned int i = 0; i < packages.size(); ++i){
		const auto infoType = m_trainingsConfig.isTreeAmountMode() ? InformationPackage::InfoType::ORF_TRAIN_FIX :
							  InformationPackage::InfoType::ORF_TRAIN;
		packages[i] = new InformationPackage(infoType, 0,
											 (int) (m_trees.size() / (Real) amountOfThreads));
		packages[i]->setStandartInformation("Train trees, thread nr: " + StringHelper::number2String(i));
		packages[i]->setTrainingsTime(trainingsTimeForPackages);
		group.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::trainInParallel, this, m_generators[i], packages[i], m_trainingsConfig.m_amountOfTrees, counterForClasses, &mutexForCounter)));
	}
	int stillOneRunning = 1;
	if(m_trainingsConfig.isTimeMode()){
		InLinePercentageFiller::setActMaxTime(m_trainingsConfig.m_seconds);
	}else if(m_trainingsConfig.isTreeAmountMode()){
		InLinePercentageFiller::setActMax(m_trainingsConfig.m_amountOfTrees);
	}else if(m_trainingsConfig.hasMemoryConstraint()){
		InLinePercentageFiller::setActMax(m_trainingsConfig.m_memory);
	}else{
		printError("The type is not defined here!");
	}
//	const unsigned long maxTime = 86400;// more than a day
//	MemoryType maxAmountOfUsedMemory;
//	Settings::getValue("OnlineRandomForest.maxAmountOfUsedMemory", maxAmountOfUsedMemory);

//	if(m_desiredAmountOfTrees == 0){
//		if(trainingsTime > maxTime){
//			InLinePercentageFiller::setActMax((long) maxAmountOfUsedMemory);
//		}else{
//			InLinePercentageFiller::setActMaxTime(trainingsTime);
//		}
//	}else{
//		InLinePercentageFiller::setActMax(m_desiredAmountOfTrees);
//	}
//	Real nextCheck = std::min(10.,m_ownSamplingTime / 10.);
	StopWatch sw;
	int lastCounter = 0;
	std::list<Real> points;
	while(stillOneRunning != 0){
		stillOneRunning = 0;
		for(unsigned int i = 0; i < packages.size(); ++i){
			if(!packages[i]->isTaskFinished()){
				++stillOneRunning;
			}
		}
		if(m_trainingsConfig.isTimeMode()){
			InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(m_trees.size());
		}else if(m_trainingsConfig.isTreeAmountMode()){
			InLinePercentageFiller::setActValueAndPrintLine(m_trees.size());
		}else if(m_trainingsConfig.hasMemoryConstraint()){
			InLinePercentageFiller::setActValueAndPrintLine((long) std::min(m_usedMemory, m_trainingsConfig.m_memory));
		}else{
			printError("The type is not defined here!");
		}
		if(counterForClasses != nullptr && m_trees.size() > 0 && m_trees.size() - lastCounter >= 1){
			lastCounter = (int) m_trees.size();
			int correct = 0;
			mutexForCounter.lock();
			for(unsigned int i = 0; i < m_storage.size(); ++i){
				if(m_storage[i]->getLabel() == argMax((*counterForClasses)[i].cbegin(), (*counterForClasses)[i].cend())){
					++correct;
				}
			}
			mutexForCounter.unlock();
			points.push_back(correct / (Real) m_storage.size() * (Real) 100.);
		}
		sleepFor(0.05);
	}
	group.join_all();
	if(m_savedAnyTreesToDisk){
		writeTreesToDisk((unsigned int) m_trees.size()); // will delete all trees fsrom memory
		loadBatchOfTreesFromDisk(0); // load first batch
	}else{
		printOnScreen("Used memory: " << StringHelper::convertMemorySpace(m_usedMemory));
	}
	printOnScreen("Calculated " << m_trees.size() << " trees with depth: " << m_maxDepth);
	if(m_trainingsConfig.isTimeMode()){
		InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(m_trees.size(), true);
	}else if(m_trainingsConfig.isTreeAmountMode()){
		InLinePercentageFiller::setActValueAndPrintLine(m_trees.size());
	}else if(m_trainingsConfig.hasMemoryConstraint()){
		InLinePercentageFiller::setActValueAndPrintLine((long) m_trainingsConfig.m_memory);
	}else{
		printError("The type is not defined here!");
	}
	for(unsigned int i = 0; i < packages.size(); ++i){
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

void OnlineRandomForest::writeTreesToDisk(const unsigned int amountOfTrees) const{
	if(amountOfTrees > 1){
		m_treesMutex.lock();
		// two different files are needed
		const std::string fileFirst = m_folderForSavedTrees + "trees_" + StringHelper::number2String(m_savedToDiskTreesFilePaths.size()) + "_firstHalf.binary";
		const std::string fileSecond = m_folderForSavedTrees + "trees_" + StringHelper::number2String(m_savedToDiskTreesFilePaths.size()) + "_secondHalf.binary";
		m_savedToDiskTreesFilePaths.push_back(std::pair<std::string, std::string>(fileFirst, fileSecond));
		std::string file = fileFirst;
		unsigned int start = 0;
		unsigned int end = amountOfTrees / 2;
		for(unsigned int j = 0; j < 2; ++j){
			std::fstream output;
			output.open(file, std::fstream::out | std::fstream::trunc);
			if(output.is_open()){
				const unsigned int len = end - start;
				output.write((char*) (&len), sizeof(unsigned int));
				for(unsigned int i = start; i < end; ++i){
					DynamicDecisionTreeInterface* tree = m_trees.front();
					m_trees.pop_front();
					if(m_useBigDynamicDecisionTrees){
						auto* dtTree = dynamic_cast<BigDynamicDecisionTree*>(tree);
						ReadWriterHelper::writeBigDynamicTree(output, *dtTree);
						SAVE_DELETE(dtTree);
					}else{
						auto* dtTree = dynamic_cast<DynamicDecisionTree<unsigned int>*>(tree);
						ReadWriterHelper::writeDynamicTree<unsigned int>(output, *dtTree);
						SAVE_DELETE(dtTree);
					}
				}
			}else{
				printError("This file could not be opened: " << file);
			}
			output.close();
			file = fileSecond;
			start = end;
			end = amountOfTrees;
		}
		m_usedMemory = 0;
		m_treesMutex.unlock();
	}
}

void OnlineRandomForest::loadBatchOfTreesFromDisk(const unsigned int batchNr) const{
	if(m_trees.size() == 0){
		if(batchNr < m_savedToDiskTreesFilePaths.size()){
			m_treesMutex.lock();
			std::string file = m_savedToDiskTreesFilePaths[batchNr].first;
			for(unsigned int j = 0; j < 2; ++j){
				std::fstream output;
				output.open(file);
				if(output.is_open()){
					unsigned int amountOfTrees = 0;
					output.read((char*) (&amountOfTrees), sizeof(unsigned int));
					for(unsigned int i = 0; i < amountOfTrees; ++i){
						if(m_useBigDynamicDecisionTrees){
							auto* newTree = new BigDynamicDecisionTree(m_storage);
							ReadWriterHelper::readBigDynamicTree(output, *newTree);
							m_trees.push_back((DynamicDecisionTreeInterface*) newTree);
							m_usedMemory += newTree->getMemSize();
						}else{
							auto* newTree = new DynamicDecisionTree<unsigned int>(m_storage);
							ReadWriterHelper::readDynamicTree(output, *newTree);
							m_trees.push_back((DynamicDecisionTreeInterface*) newTree);
							m_usedMemory += newTree->getMemSize();
						}
					}
				}
				file = m_savedToDiskTreesFilePaths[batchNr].second;
			}
			m_treesMutex.unlock();
		}
	}else{
		printError("The trees size is not zero this could be dangerous!");
	}
}

void OnlineRandomForest::tryAmountForLayers(RandomNumberGeneratorForDT* generator, const Real secondsPerSplit,
											std::list<std::pair<unsigned int, unsigned int> >* layerValues,
											boost::mutex* mutex, std::pair<int, int>* bestLayerSplit,
											Real* bestCorrectness){
	while(true){
		mutex->lock();
		if(layerValues->size() > 0){
			const int layerAmount = layerValues->front().first;
			const int amountOfFastLayers = layerValues->front().second;
			layerValues->pop_front();
			mutex->unlock();
			StopWatch sw;
			auto counter = 0u;
			while(sw.elapsedSeconds() < secondsPerSplit){
				// will remove the tree after training -> free memory
				auto tree = std::make_unique<BigDynamicDecisionTree>(m_storage, m_maxDepth, m_amountOfClasses,
																	 layerAmount, amountOfFastLayers,
																	 m_amountOfPointsCheckedPerSplit);
				tree->train((unsigned int) m_amountOfUsedDims, *generator);
				++counter;
			}
			const Real corr = counter; //correctAmount / (Real) m_storage.size() * 100. ;
			mutex->lock();
			printOnScreen("Test: " << layerAmount << ", " << amountOfFastLayers << ", with " << corr << " trees");
			if(corr > *bestCorrectness || (corr >= *bestCorrectness && layerAmount > bestLayerSplit->first)){
				bestLayerSplit->first = layerAmount;
				bestLayerSplit->second = amountOfFastLayers;
				*bestCorrectness = corr;
				printOnScreen("New best layer amount: " << layerAmount << ", " << amountOfFastLayers << ", with " << *bestCorrectness << " trees");
			}
			mutex->unlock();
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
		case OnlineStorage<LabeledVectorX*>::Event::APPEND:{
			++m_counterForRetrain;
			if(m_counterForRetrain >= m_amountOfPointsUntilRetrain){
				update();
				m_counterForRetrain = 0;
			}
			break;
		}
		case OnlineStorage<LabeledVectorX*>::Event::APPENDBLOCK:{
			update();
			m_counterForRetrain = 0;
			break;
		}
		case OnlineStorage<LabeledVectorX*>::Event::ERASE:{
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
		auto list = std::make_unique<SortedDecisionTreeList>(); // new SortedDecisionTreeList());
		printOnScreen("Predict all trees on all data points and sort them");
		sortTreesAfterPerformance(*list);
		printOnScreen("Finished sorting, worst tree has: " << list->begin()->second);

		if(list->size() != m_trees.size()){
			printError("The sorting process failed, list size is: " << list->size() << ", should be: " << m_trees.size());
			return false;
		}
		StopWatch swWhole;
		const auto nrOfParallel = (unsigned int) std::min((int) ThreadMaster::getAmountOfThreads(), (int) m_trees.size());
		const Real trainingsTimeForPackages = m_trainingsConfig.isTimeMode() ? m_trainingsConfig.m_seconds : 0;
		auto counter = 0u;
		const auto amountOfElements = (unsigned int) m_trees.size() / nrOfParallel;
		boost::mutex mutexForCounter;
		std::vector<InformationPackage*> packages(nrOfParallel, nullptr);
		boost::thread_group group;
		const auto infoType = m_trainingsConfig.isTimeMode() ? InformationPackage::InfoType::ORF_TRAIN : InformationPackage::InfoType::ORF_TRAIN_FIX;
		for(unsigned int i = 0; i < packages.size(); ++i){
			packages[i] = new InformationPackage(infoType, 0,
												 (int) (m_trees.size() / (Real) nrOfParallel));
			packages[i]->setStandartInformation("Train trees, thread nr: " + StringHelper::number2String(i));
			packages[i]->setTrainingsTime(trainingsTimeForPackages);
			group.add_thread(new boost::thread(
					boost::bind(&OnlineRandomForest::updateInParallel, this, list.get(),
								amountOfElements, &mutexForCounter, i, packages[i], &counter)));
		}
		int stillOneRunning = 1;
		if(m_trainingsConfig.isTimeMode()){
			InLinePercentageFiller::setActMaxTime(m_trainingsConfig.m_seconds);
		}else if(m_trainingsConfig.isTreeAmountMode()){
			InLinePercentageFiller::setActMax(m_trainingsConfig.m_amountOfTrees);
		}else if(m_trainingsConfig.hasMemoryConstraint()){
			InLinePercentageFiller::setActMax(m_trees.size());
		}else{
			printError("The type is not defined here!");
		}
		StopWatch sw;
		int lastCounter = 0;
		std::list<Real> points;
		while(stillOneRunning != 0){
			stillOneRunning = 0;
			for(unsigned int i = 0; i < packages.size(); ++i){
				if(!packages[i]->isTaskFinished()){
					++stillOneRunning;
				}
			}
			mutexForCounter.lock();
			if(m_trainingsConfig.isTimeMode()){
				InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(counter);
			}else if(m_trainingsConfig.isTreeAmountMode()){
				InLinePercentageFiller::setActValueAndPrintLine(counter);
			}else if(m_trainingsConfig.hasMemoryConstraint()){
				InLinePercentageFiller::setActValueAndPrintLine(counter);
			}else{
				printError("The type is not defined here!");
			}
			mutexForCounter.unlock();
			sleepFor(0.05);
		}
		group.join_all();
		printOnScreen("Updating finished, took: " << swWhole.elapsedAsTimeFrame());
		if(m_trainingsConfig.isTimeMode()){
			InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(m_trees.size(), true);
		}else if(m_trainingsConfig.isTreeAmountMode()){
			InLinePercentageFiller::setActValueAndPrintLine(m_trees.size());
		}else if(m_trainingsConfig.hasMemoryConstraint()){
			InLinePercentageFiller::setActValueAndPrintLine((long) m_trainingsConfig.m_memory);
		}else{
			printError("The type is not defined here!");
		}
		for(auto& package : packages){
			ThreadMaster::threadHasFinished(package);
			SAVE_DELETE(package);
		}
		m_trees.clear(); // the trees are not longer valid -> so removing the pointer is no problem
		for(auto& ele : *list){
			m_trees.push_back(ele.first);
		}
		printOnScreen("New worst tree has: " << list->begin()->second);
	}
	return true;
}

void OnlineRandomForest::sortTreesAfterPerformance(SortedDecisionTreeList& list){
	const unsigned int nrOfParallel = std::min(ThreadMaster::getAmountOfThreads(), (unsigned int) m_trees.size());
	boost::mutex read, append;
	boost::thread_group group;
	DecisionTreesContainer copyOfTrees;
	std::vector<InformationPackage*> packages(nrOfParallel, nullptr);
	copyOfTrees.insert(copyOfTrees.begin(), m_trees.begin(), m_trees.end());
	for(unsigned int i = 0; i < nrOfParallel; ++i){
		packages[i] = new InformationPackage(InformationPackage::InfoType::ORF_TRAIN, (Real) 0., (int) (m_trees.size() / 8));
		group.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::sortTreesAfterPerformanceInParallel, this, &list, &copyOfTrees, &read, &append, packages[i])));
	}
	group.join_all();
	for(unsigned int i = 0; i < nrOfParallel; ++i){
		ThreadMaster::threadHasFinished(packages[i]);
		SAVE_DELETE(packages[i]);
	}
}

void OnlineRandomForest::sortTreesAfterPerformanceInParallel(SortedDecisionTreeList* list, DecisionTreesContainer* trees,
		boost::mutex* readMutex, boost::mutex* appendMutex, InformationPackage* package){
	package->setStandartInformation("Sort trees after performance");
	if(m_trees.size() == 1){
		auto correct = 0u;
		for(auto& point : m_storage){
			if(point->getLabel() == (*m_trees.begin())->predict(*point)){
				++correct;
			}
		}
		list->push_back(SortedDecisionTreePair(*m_trees.begin(), correct / (Real) (m_storage.size()) * 100));
		return;
	}
	ThreadMaster::appendThreadToList(package);
	package->wait();
	SortedDecisionTreeList ownList;
	auto usedTrees = (unsigned int) (trees->size() / 16);
	while(trees->size() > 0){
		ownList.clear();
		readMutex->lock();
		for(unsigned int i = 0; i < usedTrees; ++i){
			if(trees->size() > 0){
				ownList.push_back(SortedDecisionTreePair(trees->back(), 0));
				trees->pop_back();
			}
		}
		const auto treeAmount = (unsigned int) trees->size();
		readMutex->unlock();
		printInPackageOnScreen(package, "Predict new tree, rest amount is: " << treeAmount);
		for(auto it = ownList.begin(); it != ownList.end(); ++it){
			int correct = 0;
			DynamicDecisionTreeInterface* tree = it->first;
//			for(unsigned int k = m_storage.getLastUpdateIndex(); k < m_storage.size(); ++k){
//				const LabeledVectorX& point = *(m_storage[k]);
			for(OnlineStorage<LabeledVectorX*>::ConstIterator itPoint = m_storage.begin(); itPoint != m_storage.end(); ++itPoint){
				LabeledVectorX& point = **itPoint;
				if(point.getLabel() == tree->predict(point)){
					++correct;
				}
			}
			it->second = correct / (Real) (m_storage.size()) * (Real) 100.;
		}
		SortedDecisionTreeList sortedList;
		for(auto it = ownList.begin(); it != ownList.end(); ++it){
			if(sortedList.size() == 0){
				sortedList.push_back(*it);
			}else{
				bool append = false;
				for(auto itSort = sortedList.begin(); itSort != sortedList.end(); ++itSort){
					if(itSort->second > it->second){
						sortedList.insert(itSort, *it);
						append = true;
						break;
					}
				}
				if(!append){
					sortedList.push_back(*it);
				}
			}
		}
		appendMutex->lock();
		mergeSortedLists(list, &sortedList);
		appendMutex->unlock();
		if(package->shouldThreadBePaused()){
			package->wait();
		}
		// can not be broken!
	}
	package->finishedTask();
}

void OnlineRandomForest::updateInParallel(SortedDecisionTreeList* list, const unsigned int amountOfSteps, boost::mutex* mutex, unsigned int threadNr, InformationPackage* package, unsigned int* counter){
	if(package == nullptr){
		printError("This thread has no valid information package: " + StringHelper::number2String(threadNr));
		return;
	}
	package->setStandartInformation("Orf updating thread Nr: " + StringHelper::number2String(threadNr));
	ThreadMaster::appendThreadToList(package);
	package->wait();
	mutex->lock();
	SortedDecisionTreePair pair = *list->begin(); // copy of the first element
	list->pop_front(); // remove it
	mutex->unlock();
	DynamicDecisionTreeInterface* switcher = nullptr;
	if(m_useBigDynamicDecisionTrees){
		switcher = new BigDynamicDecisionTree(m_storage, m_maxDepth, m_amountOfClasses, m_amountOfUsedLayer.first, m_amountOfUsedLayer.second, m_amountOfPointsCheckedPerSplit);
	}else{
		switcher = new DynamicDecisionTree<unsigned int>(m_storage, m_maxDepth, m_amountOfClasses, m_amountOfPointsCheckedPerSplit);
	}
	Real correctValOfSwitcher = pair.second;
	for(unsigned int i = 0; i < amountOfSteps - 1; ++i){
		switcher->train((unsigned int) m_amountOfUsedDims, *m_generators[threadNr]); // retrain worst tree
		int correct = 0;
		for(const auto& point : m_storage){
			if(point->getLabel() == switcher->predict(*point)){
				++correct;
			}
		}
		const auto correctVal = correct / (Real) m_storage.size() * (Real) 100.;
		mutex->lock();
		pair = std::move(*list->begin()); // get new element
		list->pop_front(); // remove it
		mutex->unlock();
		DynamicDecisionTreeInterface* addToList = nullptr;
		Real usedCorrectVal;
		if(correctVal > pair.second){
			// if the switcher performs better than the original tree, change both so
			// that the switcher (with the better result is placed in the list), and the original element gets the new switcher
			addToList = switcher;
			switcher = pair.first;
			usedCorrectVal = correctVal;
			correctValOfSwitcher = pair.second; // value of the switcher
			// add to list again!
			printInPackageOnScreen(package, "Performed new step with better correctness of: " << StringHelper::number2String(correctVal, 2) << " %%, worst had: " << pair.second);
		}else{
			addToList = pair.first;
			usedCorrectVal = pair.second;
			correctValOfSwitcher = correctVal;
			// no switch -> the switcher is trys to improve itself
			printInPackageOnScreen(package, "Performed new step with worse correctness of " << StringHelper::number2String(correctVal, 2) << " %% not used, worst had: " << pair.second);
		}
		mutex->lock();
		*counter += 1; // is already protected in mutex lock
		internalAppendToSortedList(list, addToList, usedCorrectVal); // insert decision tree again in the list
		mutex->unlock();
		if(package->shouldThreadBePaused()){
			package->wait();
		}else if(package->shouldThreadBeAborted()){
			break;
		}
	}
	mutex->lock();
	*counter += 1; // is already protected in mutex lock
	internalAppendToSortedList(list, switcher, correctValOfSwitcher); // insert decision tree again in the list
	mutex->unlock();
	package->finishedTask();
}

void OnlineRandomForest::internalAppendToSortedList(SortedDecisionTreeList* list, DynamicDecisionTreeInterface* pTree, Real correctVal){
	if(m_savedAnyTreesToDisk){
		printError("Not all trees used!");
		return;
	}
	if(list->size() == 0){
		list->emplace_back(pTree, correctVal);
	}else{
		bool added = false;
		for(auto it = list->begin(); it != list->end(); ++it){
			if(it->second > correctVal){
				list->emplace(it, pTree, correctVal);
				added = true;
				break;
			}
		}
		if(!added){
			list->emplace_back(pTree, correctVal);
		}
	}
}

void OnlineRandomForest::mergeSortedLists(SortedDecisionTreeList* aimList, SortedDecisionTreeList* other){
	auto itOther = other->begin();
	auto it = aimList->begin();
	while(it != aimList->end() && itOther != other->end()){
		if(itOther->second < it->second){
			it = aimList->insert(it, *itOther);
			++itOther;
		}
		++it;
	}
	while(itOther != other->end()){
		aimList->push_back(*itOther);
		++itOther;
	}
}

OnlineRandomForest::DecisionTreeIterator OnlineRandomForest::findWorstPerformingTree(Real& correctAmount){
	if(m_savedAnyTreesToDisk){
		printError("Not all trees used!");
		return m_trees.end();
	}
	int minCorrect = m_storage.size();
	auto itWorst = m_trees.end();
	for(DecisionTreeIterator itTree = m_trees.begin(); itTree != m_trees.end(); ++itTree){
		int correct = 0;
		DynamicDecisionTreeInterface* tree = *itTree;
		for(OnlineStorage<LabeledVectorX*>::ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
			LabeledVectorX& point = *(*it);
			if(point.getLabel() == tree->predict(point)){
				++correct;
			}
		}
		if(minCorrect > correct){
			minCorrect = correct;
			itWorst = itTree;
		}
	}
	correctAmount = minCorrect / (Real) m_storage.size() * (Real) 100.;
	printOnScreen("Worst correct is: " << correctAmount);
	return itWorst;
}

unsigned int OnlineRandomForest::predict(const VectorX& point) const {
	if(m_savedAnyTreesToDisk){
		printError("Not all trees used!");
		return 0;
	}
	if(m_firstTrainingDone){
		std::vector<unsigned int> values(m_amountOfClasses, 0u);
		for(auto& tree : m_trees){
			++values[tree->predict(point)];
		}
		return (unsigned int) argMax(values.cbegin(), values.cend());
	}
	return UNDEF_CLASS_LABEL;
}

Real OnlineRandomForest::predict(const VectorX& point1, const VectorX& point2, const unsigned int sampleAmount) const{
	if(m_savedAnyTreesToDisk){
			printError("Not all trees used!");
			return 0;
	}
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
			return inSameClassCounter / (Real) std::min(sampleAmount, (unsigned int) m_trees.size());
	}
	return -1;
}

Real OnlineRandomForest::predictPartitionEquality(const VectorX& point1, const VectorX& point2, RandomUniformNr& uniformNr, unsigned int amountOfSamples) const{
	if(m_savedAnyTreesToDisk){
		printError("Not all trees used!");
		return 0;
	}
	if(m_firstTrainingDone && m_trees.size() > 0){
		if(amountOfSamples > (unsigned int) m_trees.size()){
			amountOfSamples = (unsigned int)  m_trees.size();
		}
		if(m_maxDepth <= 3 || amountOfSamples == 0){
			printError("Amount of samples: " << amountOfSamples << ", tree size: " << m_trees.size() << ", max depth: " << m_maxDepth);
			return -1;
		}
		int sameLeaveCounter = 0;
		auto counter = 0u;
		for(auto it = m_trees.cbegin(); it != m_trees.cend() && counter < amountOfSamples; ++it){
			const auto height = (unsigned int) uniformNr();
			if((*it)->predictIfPointsShareSameLeaveWithHeight(point1, point2, height)){
				++sameLeaveCounter;
			}
			++counter;
		}
		return sameLeaveCounter / (Real) std::min(amountOfSamples, (unsigned int) m_trees.size());
	}
	return -1;
}

void OnlineRandomForest::predictData(const Data& points, Labels& labels) const{
	if(m_savedAnyTreesToDisk){
		std::vector<std::vector<Real> > probs;
		predictData(points, labels, probs);
	}else{
		labels.resize(points.size());
		boost::thread_group group;
		const auto nrOfParallel = ThreadMaster::getAmountOfThreads();
		std::vector<InformationPackage*> packages(nrOfParallel, nullptr);
		for(unsigned int i = 0; i < nrOfParallel; ++i){
			auto start = (const int) (i / (Real) nrOfParallel * points.size());
			auto end = (const int) ((i + 1) / (Real) nrOfParallel * points.size());
			packages[i] = new InformationPackage(InformationPackage::ORF_PREDICT, 0, end - start);
			packages[i]->setStandartInformation("Thread for orf prediction: " + StringHelper::number2String(i + 1));
			group.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::predictDataInParallel, this, points, &labels, packages[i], start, end)));
		}
		bool stillOneRunning = true;
		InLinePercentageFiller::setActMax(points.size());
		while(stillOneRunning){
			stillOneRunning = false;
			long counter = 0;
			for(auto& package : packages){
				if(!package->isTaskFinished()){
					stillOneRunning = true;
				}
				counter += package->amountOfTrainingStepsPerformed(); // corresponds to amount of points predicted
			}
			InLinePercentageFiller::setActValueAndPrintLine(counter);
			sleepFor(0.1);
		}
		group.join_all();
	}
}

void OnlineRandomForest::predictData(const LabeledData& points, Labels& labels) const{
	if(m_savedAnyTreesToDisk){
		std::vector<std::vector<Real> > probs;
		predictData(points, labels, probs);
	}else{
		labels.resize(points.size());
		boost::thread_group group;
		const unsigned int nrOfParallel = ThreadMaster::getAmountOfThreads();
		std::vector<InformationPackage*> packages(nrOfParallel, nullptr);
		for(unsigned int i = 0; i < nrOfParallel; ++i){
			auto start = (const int) (i / (Real) nrOfParallel * points.size());
			auto end = (const int) ((i + 1) / (Real) nrOfParallel * points.size());
			packages[i] = new InformationPackage(InformationPackage::ORF_PREDICT, 0, end - start);
			packages[i]->setStandartInformation("Thread for orf prediction: " + StringHelper::number2String(i + 1));
			group.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::predictClassDataInParallel, this, points, &labels, packages[i], start, end)));
		}
		bool stillOneRunning = true;
		InLinePercentageFiller::setActMax(points.size());
		while(stillOneRunning){
			stillOneRunning = false;
			long counter = 0;
			for(auto& package : packages){
				if(!package->isTaskFinished()){
					stillOneRunning = true;
				}
				counter += package->amountOfTrainingStepsPerformed(); // corresponds to amount of points predicted
			}
			InLinePercentageFiller::setActValueAndPrintLine(counter);
			sleepFor(0.1);
		}
		group.join_all();
	}
}

void OnlineRandomForest::predictData(const Data& points, Labels& labels, std::vector< std::vector<Real> >& probabilities) const{
	labels.resize(points.size());
	probabilities.resize(points.size());
	if(m_savedAnyTreesToDisk){
		boost::thread_group group;
		const unsigned int nrOfParallel = ThreadMaster::getAmountOfThreads();
		std::vector<std::vector<std::vector<Real> > >* probsForThreads = new std::vector<std::vector<std::vector<Real> > >(nrOfParallel);
		unsigned int batchNr = 0;
		if(m_trees.size() == 0 && m_savedAnyTreesToDisk && batchNr < m_savedToDiskTreesFilePaths.size()){
			loadBatchOfTreesFromDisk(batchNr);
			++batchNr;
		} // else means the save mode is not used
		boost::mutex mutexForTrees;
		DecisionTreeIterator it = m_trees.begin();
		for(unsigned int i = 0; i < nrOfParallel; ++i){
			std::vector<std::vector<Real> >* actProb = &((*probsForThreads)[i]);
			actProb->resize(points.size());
			for(unsigned int j = 0; j < points.size(); ++j){
				(*actProb)[j].resize(m_amountOfClasses);
			}
			group.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::predictDataProbInParallel, this, points, actProb, &batchNr, &mutexForTrees, &it)));
		}
		group.join_all();
		for(unsigned int i = 0; i < points.size(); ++i){
			unsigned int iMax = UNDEF_CLASS_LABEL;
			Real max = (Real) 0.;
			const Real fac = (Real) 1. / getNrOfTrees();
			probabilities[i].resize(m_amountOfClasses);
			for(unsigned int j = 0; j < m_amountOfClasses; ++j){
				Real val = 0;
				for(unsigned int k = 0; k < nrOfParallel; ++k){
					val += (*probsForThreads)[k][i][j];
				}
				probabilities[i][j] = val * fac;
				if(max < probabilities[i][j]){
					max = probabilities[i][j];
					iMax = j;
				}
			}
			labels[i] = iMax;
		}
		SAVE_DELETE(probsForThreads);
	}else{
		boost::thread_group group;
		const auto nrOfParallel = ThreadMaster::getAmountOfThreads();
		std::vector<InformationPackage*> packages(nrOfParallel, nullptr);
		for(unsigned int i = 0; i < nrOfParallel; ++i){
			const int start = (int) (i / (Real) nrOfParallel * points.size());
			const int end = (int) ((i + 1) / (Real) nrOfParallel * points.size());
			packages[i] = new InformationPackage(InformationPackage::ORF_PREDICT, 0, end - start);
			packages[i]->setStandartInformation("Thread for orf prediction: " + StringHelper::number2String(i + 1));
			group.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::predictDataProbInParallelStartEnd, this, points, &labels, &probabilities, packages[i], start, end)));
		}
		bool stillOneRunning = true;
		InLinePercentageFiller::setActMax(points.size());
		while(stillOneRunning){
			stillOneRunning = false;
			long counter = 0;
			for(auto& package : packages){
				if(!package->isTaskFinished()){
					stillOneRunning = true;
				}
				counter += package->amountOfTrainingStepsPerformed(); // corresponds to amount of points predicted
			}
			InLinePercentageFiller::setActValueAndPrintLine(counter);
			sleepFor(0.1);
		}
		group.join_all();
	}
}

void OnlineRandomForest::predictData(const LabeledData& points, Labels& labels, std::vector< std::vector<Real> >& probabilities) const{
	labels.resize(points.size());
	probabilities.resize(points.size());
	if(m_savedAnyTreesToDisk){
		boost::thread_group group;
		const auto nrOfParallel = ThreadMaster::getAmountOfThreads();
		auto probsForThreads = new std::vector<std::vector<std::vector<Real> > >(nrOfParallel);
		unsigned int batchNr = 0;
		if(m_trees.size() == 0 && m_savedAnyTreesToDisk && batchNr < m_savedToDiskTreesFilePaths.size()){
			loadBatchOfTreesFromDisk(batchNr);
			++batchNr;
		} // else means the save mode is not used
		boost::mutex mutexForTrees;
		auto it = m_trees.begin();
		for(unsigned int i = 0; i < nrOfParallel; ++i){
			std::vector<std::vector<Real> >* actProb = &((*probsForThreads)[i]);
			actProb->resize(points.size());
			for(unsigned int j = 0; j < points.size(); ++j){
				(*actProb)[j].resize(m_amountOfClasses);
			}
			group.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::predictClassDataProbInParallel, this, points, actProb, &batchNr, &mutexForTrees, &it)));
		}
		group.join_all();
		for(unsigned int i = 0; i < points.size(); ++i){
			unsigned int iMax = UNDEF_CLASS_LABEL;
			Real max((Real) 0.);
			const Real fac = (Real) 1. / getNrOfTrees();
			probabilities[i].resize(m_amountOfClasses);
			for(unsigned int j = 0; j < m_amountOfClasses; ++j){
				Real val = 0;
				for(unsigned int k = 0; k < nrOfParallel; ++k){
					val += (*probsForThreads)[k][i][j];
				}
				probabilities[i][j] = val * fac;
				if(max < probabilities[i][j]){
					max = probabilities[i][j];
					iMax = j;
				}
			}
			labels[i] = iMax;
		}
		SAVE_DELETE(probsForThreads);
	}else{
		boost::thread_group group;
		const auto nrOfParallel = ThreadMaster::getAmountOfThreads();
		std::vector<InformationPackage*> packages(nrOfParallel, nullptr);
		for(unsigned int i = 0; i < nrOfParallel; ++i){
			const int start = (int) (i / (Real) nrOfParallel * points.size());
			const int end = (int) ((i + 1) / (Real) nrOfParallel * points.size());
			packages[i] = new InformationPackage(InformationPackage::ORF_PREDICT, 0, end - start);
			packages[i]->setStandartInformation("Thread for orf prediction: " + StringHelper::number2String(i + 1));
			group.add_thread(new boost::thread(boost::bind(&OnlineRandomForest::predictClassDataProbInParallelStartEnd, this, points, &labels, packages[i], &probabilities, start, end)));
		}
		bool stillOneRunning = true;
		InLinePercentageFiller::setActMax(points.size());
		while(stillOneRunning){
			stillOneRunning = false;
			long counter = 0;
			for(auto& package : packages){
				if(!package->isTaskFinished()){
					stillOneRunning = true;
				}
				counter += package->amountOfTrainingStepsPerformed(); // corresponds to amount of points predicted
			}
			InLinePercentageFiller::setActValueAndPrintLine(counter);
			sleepFor(0.1);
		}
		group.join_all();
	}
}

void OnlineRandomForest::predictDataProbInParallel(const Data& points, std::vector< std::vector<Real> >* probabilities,
		unsigned int* iBatchNr, boost::mutex* mutex, DecisionTreeIterator* itOfActElement) const{
	if(m_firstTrainingDone){
			bool hasNewTrees = true;
			while(hasNewTrees){
				hasNewTrees = false;
				while(true){
					mutex->lock();
					DynamicDecisionTreeInterface* actTree = nullptr;
					if((*itOfActElement) != m_trees.end()){
						actTree = **itOfActElement;
						++(*itOfActElement);
					}
					mutex->unlock();
					if(actTree != nullptr){
						for(unsigned int i = 0; i < points.size(); ++i){
							const unsigned int label = actTree->predict(*points[i]);
							(*probabilities)[i][label] += 1;
						}
					}else{
						break;
					}
				}
				// no trees left load new trees!
				mutex->lock();
				if((*itOfActElement) == m_trees.end() && m_savedAnyTreesToDisk && *iBatchNr < m_savedToDiskTreesFilePaths.size()){ // first thread achieved that after retraining
					printOnScreen("Load batch: " << *iBatchNr);
					for(DecisionTreeIterator it = m_trees.begin(); it != m_trees.end(); ++it){
						SAVE_DELETE(*it);
					}
					m_trees.clear();
					loadBatchOfTreesFromDisk(*iBatchNr);
					*itOfActElement = m_trees.begin();
					++(*iBatchNr);
				}
				mutex->unlock();
				mutex->lock();
				if((*itOfActElement) != m_trees.end()){
					hasNewTrees = true;
				}
				mutex->unlock();
			}
		}
}

void OnlineRandomForest::predictClassDataProbInParallel(const LabeledData& points, std::vector< std::vector<Real> >* probabilities,
		unsigned int* iBatchNr, boost::mutex* mutex, DecisionTreeIterator* itOfActElement) const{
	if(m_firstTrainingDone){
		bool hasNewTrees = true;
		while(hasNewTrees){
			hasNewTrees = false;
			while(true){
				mutex->lock();
				DynamicDecisionTreeInterface* actTree = nullptr;
				if((*itOfActElement) != m_trees.end()){
					actTree = **itOfActElement;
					++(*itOfActElement);
				}
				mutex->unlock();
				if(actTree != nullptr){
					for(unsigned int i = 0; i < points.size(); ++i){
						const unsigned int label = actTree->predict(*points[i]);
						(*probabilities)[i][label] += 1;
					}
				}else{
					break;
				}
			}
			// no trees left load new trees!
			mutex->lock();
			if((*itOfActElement) == m_trees.end() && m_savedAnyTreesToDisk && *iBatchNr < m_savedToDiskTreesFilePaths.size()){ // first thread achieved that after retraining
				printOnScreen("Load batch: " << *iBatchNr);
				for(DecisionTreeIterator it = m_trees.begin(); it != m_trees.end(); ++it){
					SAVE_DELETE(*it);
				}
				m_trees.clear();
				loadBatchOfTreesFromDisk(*iBatchNr);
				*itOfActElement = m_trees.begin();
				++(*iBatchNr);
			}
			mutex->unlock();
			mutex->lock();
			if((*itOfActElement) != m_trees.end()){
				hasNewTrees = true;
			}
			mutex->unlock();
		}
	}
}

void OnlineRandomForest::predictDataProbInParallelStartEnd(const Data& points, Labels* labels, std::vector< std::vector<Real> >* probabilities, InformationPackage* package, const unsigned int start, const unsigned int end) const{
	if(m_firstTrainingDone){
		ThreadMaster::appendThreadToList(package);
		package->wait();
		printInPackageOnScreen(package, "Done: " << 0 << " %%");
		for(unsigned int i = start; i < end; ++i){
			(*probabilities)[i].resize(m_amountOfClasses);
			for(DecisionTreeConstIterator it = m_trees.cbegin(); it != m_trees.cend(); ++it){
				(*probabilities)[i][(*it)->predict(*points[i])] += 1;
			}
			unsigned int iMax = UNDEF_CLASS_LABEL;
			Real max = (Real) 0.;
			const Real fac = (Real) 1. / m_trees.size();
			for(unsigned int j = 0; j < m_amountOfClasses; ++j){
				(*probabilities)[i][j] *= fac;
				if(max < (*probabilities)[i][j]){
					max = (*probabilities)[i][j];
					iMax = j;
				}
			}
			(*labels)[i] = iMax;
			packageUpdateForPrediction(package, i, start, end);
		}
		package->finishedTask();
	}
}

void OnlineRandomForest::predictClassDataProbInParallelStartEnd(const LabeledData& points, Labels* labels, InformationPackage* package, std::vector< std::vector<Real> >* probabilities, const unsigned int start, const unsigned int end) const{
	if(m_firstTrainingDone){
		ThreadMaster::appendThreadToList(package);
		package->wait();
		printInPackageOnScreen(package, "Done: " << 0 << " %%");
		for(unsigned int i = start; i < end; ++i){
			(*probabilities)[i].resize(m_amountOfClasses);
			for(DecisionTreeConstIterator it = m_trees.cbegin(); it != m_trees.cend(); ++it){
				(*probabilities)[i][(*it)->predict(*points[i])] += 1;
			}
			unsigned int iMax = UNDEF_CLASS_LABEL;
			Real max = (Real) 0.;
			const Real fac = (Real) 1. / m_trees.size();
			for(unsigned int j = 0; j < m_amountOfClasses; ++j){
				(*probabilities)[i][j] *= fac;
				if(max < (*probabilities)[i][j]){
					max = (*probabilities)[i][j];
					iMax = j;
				}
			}
			(*labels)[i] = iMax;
			packageUpdateForPrediction(package, i, start, end);
		}
		package->finishedTask();
	}
}

void OnlineRandomForest::predictDataInParallel(const Data& points, Labels* labels, InformationPackage* package, const unsigned int start, const unsigned int end) const{
	ThreadMaster::appendThreadToList(package);
	package->wait();
	printInPackageOnScreen(package, "Done: " << 0 << " %%");
	for(unsigned int i = start; i < end; ++i){
		(*labels)[i] = predict(*points[i]);
		packageUpdateForPrediction(package, i, start, end);
	}
	package->finishedTask();
}

void OnlineRandomForest::predictClassDataInParallel(const LabeledData& points, Labels* labels, InformationPackage* package, const unsigned int start, const unsigned int end) const{
	ThreadMaster::appendThreadToList(package);
	package->wait();
	printInPackageOnScreen(package, "Done: " << 0 << " %%");
	for(unsigned int i = start; i < end; ++i){
		(*labels)[i] = predict(*points[i]);
		packageUpdateForPrediction(package, i, start, end);
	}
	package->finishedTask();
}

void OnlineRandomForest::getLeafNrFor(std::vector<int>& leafNrs){
	leafNrs = std::vector<int>(m_amountOfClasses, 0);
	for(OnlineStorage<LabeledVectorX*>::ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
		leafNrs[predict(**it)] += 1;
	}
}

OnlineStorage<LabeledVectorX*>& OnlineRandomForest::getStorageRef(){
	return m_storage;
}

const OnlineStorage<LabeledVectorX*>& OnlineRandomForest::getStorageRef() const{
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
			m_minMaxValues[i][0] = REAL_MAX;
			m_minMaxValues[i][1] = NEG_REAL_MAX;
		}
	}
	switch(event){
	case OnlineStorage<LabeledVectorX*>::APPEND:{
		LabeledVectorX& point = *m_storage.last();
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
	case OnlineStorage<LabeledVectorX*>::APPENDBLOCK:{
		const unsigned int start = m_storage.getLastUpdateIndex();
		for(unsigned int t = start; t < m_storage.size(); ++t){
			LabeledVectorX& point = *m_storage[t];
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

void OnlineRandomForest::setTrainingsMode(const OnlineRandomForest::TrainingsConfig& config){
	if(config.m_mode != OnlineRandomForest::TrainingsConfig::TrainingsMode::UNDEFINED){
		m_trainingsConfig = config;
	}else{
		readTrainingsModeFromSetting();
	}
}

void OnlineRandomForest::readTrainingsModeFromSetting(){
	m_trainingsConfig.m_amountOfTrees = 0;
	m_trainingsConfig.m_memory = 0;
	m_trainingsConfig.m_seconds = 0;
	MemoryType maxAmountOfUsedMemory;
	Settings::getValue("OnlineRandomForest.maxAmountOfUsedMemory", maxAmountOfUsedMemory);
	unsigned int amountOfTrees;
	Settings::getValue("OnlineRandomForest.amountOfTrainedTrees", amountOfTrees);
	const Real trainingsTime = CommandSettings::get_samplingAndTraining();
	if(amountOfTrees == 0){
		if(trainingsTime > 0 && maxAmountOfUsedMemory == 0){
			m_trainingsConfig.m_mode = TrainingsConfig::TrainingsMode::TIME;
			m_trainingsConfig.m_seconds = trainingsTime;
		}else if(maxAmountOfUsedMemory > 0){
			if(trainingsTime > 0){
				m_trainingsConfig.m_mode = TrainingsConfig::TrainingsMode::TIME_WITH_MEMORY;
				m_trainingsConfig.m_seconds = trainingsTime;
				m_trainingsConfig.m_memory = maxAmountOfUsedMemory;
			}else{
				m_trainingsConfig.m_mode = TrainingsConfig::TrainingsMode::MEMORY;
				m_trainingsConfig.m_memory = maxAmountOfUsedMemory;
			}
		}
	}else{
		if(trainingsTime > 0 && maxAmountOfUsedMemory == 0){ // time overrules tree amount
			m_trainingsConfig.m_mode = TrainingsConfig::TrainingsMode::TIME;
			m_trainingsConfig.m_seconds = trainingsTime;
		}else if(maxAmountOfUsedMemory > 0){
			if(trainingsTime > 0){ // time overrules tree amount, memory is added constraint
				m_trainingsConfig.m_mode = TrainingsConfig::TrainingsMode::TIME_WITH_MEMORY;
				m_trainingsConfig.m_seconds = trainingsTime;
				m_trainingsConfig.m_memory = maxAmountOfUsedMemory;
			}else{ // trees are used, memory is added constraint
				m_trainingsConfig.m_mode = TrainingsConfig::TrainingsMode::TREEAMOUNT_WITH_MEMORY;
				m_trainingsConfig.m_amountOfTrees = amountOfTrees;
				m_trainingsConfig.m_memory = maxAmountOfUsedMemory;
			}
		}else{ // just amount of trees no memory constraint
			m_trainingsConfig.m_mode = TrainingsConfig::TrainingsMode::TREEAMOUNT;
			m_trainingsConfig.m_amountOfTrees = amountOfTrees;
		}
	}
}

void OnlineRandomForest::packageUpdateForPrediction(InformationPackage* package, const unsigned int i,
													const unsigned int start,
													const unsigned int end) const{
	if(i % 5 == 0){
		package->overwriteLastLineToScreenForThisThread(
				"Done: " + StringHelper::number2String((i - start) / (Real) (end - start) * (Real) 100., 2) +
				" %%");
	}
	if(package->shouldThreadBePaused()){
		package->wait();
	}else if(package->shouldThreadBeAborted()){
		printError("The prediciton can not be aborted!");
	}
	package->performedOneTrainingStep(); // used for the InLinePercentageFiller

}
