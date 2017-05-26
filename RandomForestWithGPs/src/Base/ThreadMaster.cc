/*
 * ThreadMaster.cc
 *
 *  Created on: 22.11.2016
 *      Author: Max
 */

#include "ThreadMaster.h"
#include "Settings.h"
#include "ScreenOutput.h"
#include "CommandSettings.h"

int ThreadMaster::m_counter = 0;
unsigned int ThreadMaster::m_maxCounter = 0;
Real ThreadMaster::m_timeToSleep = 0.1;
ThreadMaster::PackageList ThreadMaster::m_waitingList;
ThreadMaster::PackageList ThreadMaster::m_runningList;
boost::thread* ThreadMaster::m_mainThread(nullptr);
boost::mutex ThreadMaster::m_mutex;
std::atomic<bool> ThreadMaster::m_keepRunning(true);
boost::mutex ThreadMaster::m_isFinished;

void ThreadMaster::start(){
	if(m_mainThread == nullptr){
		setMaxCounter();
		m_mainThread = new boost::thread(&ThreadMaster::run);
	}
}

void ThreadMaster::setFrequence(const Real frequence){
	m_timeToSleep = std::max(Real(1.) / frequence, (Real) 0.001);
}

void ThreadMaster::threadHasFinished(InformationPackage* package){
	m_mutex.lock();
	package->finishedTask();
	m_mutex.unlock();
	bool found;
	do{
		found = false;
		m_mutex.lock();
		for(auto& waitingPackage : m_waitingList){
			if(waitingPackage == package){
				found = true;
				break;
			}
		}

		for(auto& runningPackage : m_runningList){
			if(runningPackage == package){
				found = true;
				break;
			}
		}
		m_mutex.unlock();
	}while(found); // until it is not found anymore
}

void ThreadMaster::run(){
	int nrOfInducingPoints;
	Settings::getValue("IVM.nrOfInducingPoints", nrOfInducingPoints);
//	const int amountOfPointsNeededForIvms = nrOfInducingPoints * 1.2;
	m_isFinished.lock();
	while(m_keepRunning){
		m_mutex.lock();
//		if(m_counter < m_maxCounter){
		Real bestAttractionLevel = 0;
		int minAmountOfPoints = INT_MAX, maxAmountOfPoints = 0;
		for(auto it = m_waitingList.begin(); it != m_waitingList.end(); ++it){
			const int amount = (*it)->amountOfAffectedPoints();
			if(minAmountOfPoints > amount){
				minAmountOfPoints = amount;
			}
			if(maxAmountOfPoints < amount){
				maxAmountOfPoints = amount;
			}
			if(amount == 1){
				printInPackageOnScreen(*it, "This thread has only 1 element");
			}
		}
		sortWaitingList(minAmountOfPoints, maxAmountOfPoints);
		while(m_counter < m_maxCounter && m_waitingList.size() > 0){
			auto selectedValue = m_waitingList.begin();
			if(selectedValue != m_waitingList.end()){
//				std::cout << "A thread was added to running!" << std::endl;
				m_runningList.push_back(*selectedValue); // first add to the running list
				++m_counter; // increase the counter of running threads
				while(!(*selectedValue)->isWaiting()){ // if the thread is not waiting wait until it waits for reactivation -> should happen fast
					sleepFor(0.05);
				}
				(*selectedValue)->notify(); // start running of the thread
				printInPackageOnScreen(*selectedValue, "This thread was selected in the beginning!");
				m_waitingList.erase(selectedValue); // remove the selected value out of the waiting thread list
				selectedValue = m_waitingList.end();
			}
		}
		auto selectedValue = m_waitingList.begin();
		for(auto it = m_runningList.begin(); it != m_runningList.end(); ++it){
			// for each running element check if execution is finished
			const int maxTrainingsTime = (int) ((*it)->getMaxTrainingsTime() > 0 ? (*it)->getMaxTrainingsTime() : CommandSettings::get_samplingAndTraining());
			if((*it)->getWorkedAmountOfSeconds() > maxTrainingsTime * 0.05 || (*it)->isTaskFinished()){ // each training have to take at least 5 seconds!
				if((*it)->getWorkedAmountOfSeconds() > maxTrainingsTime && !(*it)->shouldTrainingBeAborted() && (*it)->canBeAbortedAfterCertainTime()){
//					std::cout << "Abort training, has worked: " << (*it)->getWorkedAmountOfSeconds() << std::endl;
					printInPackageOnScreen(*it, "Abort training in thread master");
					(*it)->abortTraing(); // break the training
				}
				if(selectedValue != m_waitingList.end()){
					if(!(*it)->shouldTrainingBePaused() && !(*it)->isWaiting() && (*it)->runningTimeSinceLastWait() > 2.0){ // only change thread if it is running more than 2 seconds
						if((int) bestAttractionLevel > (int) (*it)->calcAttractionLevel(minAmountOfPoints, maxAmountOfPoints) || (*selectedValue)->getPriority() > (*it)->getPriority()){ // must be at least 1. point be better
							// hold this training and start the other one
							(*it)->pauseTheTraining(); // pause the actual training
							++selectedValue; // take the next one and compare that to the rest
						}
					}
				}
				if((*it)->isWaiting()){
					// there is a running thread which waits -> put him back in the waiting list
//					std::cout << "A thread was moved from waiting to paused!" << std::endl;
					auto copyIt = it;
					m_waitingList.push_back(*it); // append at the waiting list
					--it; // go one back, in the end of the loop the next element will be taken
					m_runningList.erase(copyIt); // erase the copied element
					--m_counter; // -> now a new thread can run
					continue; // without continue the first element could be made to the "zero" element, which does not exists -> seg fault
				}
				if((*it)->isTaskFinished()){
//					std::cout << "A thread is finished!" << std::endl;
					auto copyIt = it; // perform copy
					--it; // go one back, in the end of the loop the next element will be taken
					m_runningList.erase(copyIt); // erase the copied element
					// decrease the counter
					--m_counter; // -> now a new thread can run
					continue;
				}
//				if(m_waitingList.size() > 0){ // there are elements waiting in the queue
//
//				}
			}
		}
		m_mutex.unlock();
		sleepFor(m_timeToSleep);
	}
	m_isFinished.unlock();
}

void ThreadMaster::sortWaitingList(const int minAmountOfPoints, const int maxAmountOfPoints){
	if(m_waitingList.size() > 1){
		for(unsigned int k = 0; k < m_waitingList.size(); ++k){
			bool somethingChange = false;
			for(auto itPrev = m_waitingList.begin(), it = ++itPrev; it != m_waitingList.end(); ++it, ++itPrev){
				bool swap = false;
				if(!(*itPrev)->isWaiting() || (*itPrev)->getPriority() > (*it)->getPriority()){
					swap = true;
					somethingChange = true;
				}else{
					switch((*it)->getType()){
					case InfoType::IVM_INIT_DIFFERENCE_MATRIX:
						// only after priority
						break;
					case InfoType::IVM_TRAIN:
					case InfoType::IVM_RETRAIN: // has another priority -> but rest is the same
					case InfoType::IVM_MULTI_UPDATE:
					case InfoType::ORF_TRAIN:
					case InformationPackage::ORF_TRAIN_FIX:
						if((*itPrev)->calcAttractionLevel(minAmountOfPoints, maxAmountOfPoints) > (*it)->calcAttractionLevel(minAmountOfPoints, maxAmountOfPoints)){
							swap = true;
						}
						break;
					case InfoType::IVM_PREDICT:{
						const int diff = (*itPrev)->amountOfTrainingStepsPerformed() - (*it)->amountOfTrainingStepsPerformed();
						const int amountOfPoints = (int) (
								((*it)->amountOfAffectedPoints() + (*itPrev)->amountOfAffectedPoints()) * 0.5);
						if(diff > 0.1 * amountOfPoints){
							swap = true;
						}
						break;
					}default:
						printError("This type is not supported here!");
						break;
					}
				}
				if(swap){
					InformationPackage* temp = *itPrev;
					*itPrev = *it;
					*it = temp;
					somethingChange = true;
				}
			}
			if(!somethingChange){
				break;
			}
		}
	}
}

bool ThreadMaster::appendThreadToList(InformationPackage* package){
	bool ret;
	m_mutex.lock();
	m_waitingList.push_back(package);
	ret = true;
	m_mutex.unlock();
	return ret;
}

void ThreadMaster::abortAllThreads(){
	m_mutex.lock();
	for(auto& waitingPackage : m_waitingList){
		if(waitingPackage->canBeAbortedInGeneral()){
			waitingPackage->abortTraing();
		}
	}
	for(auto& runningPackage : m_runningList){
		if(runningPackage->canBeAbortedInGeneral()){
			runningPackage->abortTraing();
		}
	}
	m_mutex.unlock();
}

void ThreadMaster::setMaxCounter(){
	if(Settings::getDirectBoolValue("ThreadMaster.useMultiThread")){
		m_maxCounter = boost::thread::hardware_concurrency();
	}else{
		m_maxCounter = 1;
	}
}

const unsigned int ThreadMaster::getAmountOfThreads(){
	return m_maxCounter;
}
