/*
 * ThreadMaster.cc
 *
 *  Created on: 22.11.2016
 *      Author: Max
 */

#include "ThreadMaster.h"
#include "Settings.h"
#include "CommandSettings.h"

int ThreadMaster::m_counter = 0;
int ThreadMaster::m_maxCounter = 0;
double ThreadMaster::m_timeToSleep = 0.1;
ThreadMaster::PackageList ThreadMaster::m_waitingList;
ThreadMaster::PackageList ThreadMaster::m_runningList;
boost::thread* ThreadMaster::m_mainThread = nullptr;
boost::mutex ThreadMaster::m_mutex;


ThreadMaster::ThreadMaster() {
	// TODO Auto-generated constructor stub

}

ThreadMaster::~ThreadMaster() {
	// TODO Auto-generated destructor stub
}


void ThreadMaster::start(){
	if(m_mainThread == nullptr){
		setMaxCounter();
		m_mainThread = new boost::thread(&ThreadMaster::run);
	}
}

void ThreadMaster::setFrequence(const double frequence){
	m_timeToSleep = std::max(1. / frequence, 0.001);
}

void ThreadMaster::threadHasFinished(InformationPackage* package){
	m_mutex.lock();
	package->finishedTask();
	m_mutex.unlock();
	bool found = false;
	do{
		found = false;
		m_mutex.lock();
		for(PackageList::const_iterator it = m_waitingList.begin(); it != m_waitingList.end(); ++it){
			if(*it == package){
				found = true;
				break;
			}
		}
		for(PackageList::const_iterator it = m_runningList.begin(); it != m_runningList.end(); ++it){
			if(*it == package){
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
	const int amountOfPointsNeededForIvms = nrOfInducingPoints * 1.2;
	while(true){
		m_mutex.lock();
//		if(m_counter < m_maxCounter){
		PackageList::const_iterator selectedValue = m_waitingList.end();
		double bestAttractionLevel = 0;
		int minAmountOfPoints = INT_MAX, maxAmountOfPoints = 0;
		for(PackageList::const_iterator it = m_waitingList.begin(); it != m_waitingList.end(); ++it){
			const int amount = (*it)->amountOfAffectedPoints();
			if(minAmountOfPoints > amount){
				minAmountOfPoints = amount;
			}
			if(maxAmountOfPoints < amount){
				maxAmountOfPoints = amount;
			}
			if(amount == 1){
				(*it)->printLineToScreenForThisThread("This thread has only 1 element");
			}
		}
		for(PackageList::const_iterator it = m_waitingList.begin(); it != m_waitingList.end(); ++it){
			if(!(*it)->isWaiting()){
				continue; // hasn't reached the waiting point so the thread should be added to the running list
			}

			switch((*it)->getType()){
			case InfoType::IVM_TRAIN:
			case InfoType::IVM_RETRAIN: // has another priority -> but rest is the same
			case InfoType::IVM_MULTI_UPDATE:
				if(selectedValue == m_waitingList.end()){
					if((*it)->amountOfAffectedPoints() > amountOfPointsNeededForIvms){
						selectedValue = it;
						bestAttractionLevel = (*it)->calcAttractionLevel(minAmountOfPoints, maxAmountOfPoints);
					}
				}else{
					if((*selectedValue)->getPriority() < (*it)->getPriority()){
						const double attractionLevel = (*it)->calcAttractionLevel(minAmountOfPoints, maxAmountOfPoints);
						if(attractionLevel > bestAttractionLevel){
							bestAttractionLevel = attractionLevel;
							selectedValue = it;
						}
					}
				}
				break;
			case InfoType::ORF_TRAIN:
			case InformationPackage::ORF_TRAIN_FIX:
				if(selectedValue == m_waitingList.end()){
					selectedValue = it;
					bestAttractionLevel = (*it)->calcAttractionLevel(minAmountOfPoints, maxAmountOfPoints);
				}else{
					if((*selectedValue)->getPriority() < (*it)->getPriority()){
						const double attractionLevel = (*it)->calcAttractionLevel(minAmountOfPoints, maxAmountOfPoints);
						if(attractionLevel > bestAttractionLevel){
							bestAttractionLevel = attractionLevel;
							selectedValue = it;
						}
					}
				}
				break;
			case InfoType::IVM_PREDICT:
				if(selectedValue == m_waitingList.end()){
					selectedValue = it;
				}else{
					if((*selectedValue)->getPriority() < (*it)->getPriority()){
						const int diff = (*selectedValue)->amountOfTrainingStepsPerformed() - (*it)->amountOfTrainingStepsPerformed();
						const int amountOfPoints = ((*it)->amountOfAffectedPoints() + (*selectedValue)->amountOfAffectedPoints()) * 0.5;
						if(diff > 0.1 * amountOfPoints){
							selectedValue = it;
						}
					}
				}
				break;
			default:
				printError("This type is not supported here!");
				break;
			}
		}
		bool selectedValueWasUsed = false;
		if(m_counter < m_maxCounter){
			if(selectedValue != m_waitingList.end()){
//				std::cout << "A thread was added to running!" << std::endl;
				selectedValueWasUsed = true;
				m_runningList.push_back(*selectedValue); // first add to the running list
				++m_counter; // increase the counter of running threads
				while(!(*selectedValue)->isWaiting()){ // if the thread is not waiting wait until it waits for reactivation -> should happen fast
					usleep(0.05 * 1e6);
				}
				(*selectedValue)->notify(); // start running of the thread
				(*selectedValue)->printLineToScreenForThisThread("This thread was selected in the beginning!");
				m_waitingList.erase(selectedValue); // remove the selected value out of the waiting thread list
				selectedValue = m_waitingList.end();
			}
		}
		for(PackageList::const_iterator it = m_runningList.begin(); it != m_runningList.end(); ++it){
			// for each running element check if execution is finished
			const int maxTrainingsTime = (*it)->getMaxTrainingsTime() > 0 ? (*it)->getMaxTrainingsTime() : CommandSettings::get_samplingAndTraining();
			if((*it)->getWorkedAmountOfSeconds() > maxTrainingsTime * 0.05 || (*it)->isTaskFinished()){ // each training have to take at least 5 seconds!
				if((*it)->getWorkedAmountOfSeconds() > maxTrainingsTime && !(*it)->shouldTrainingBeAborted() && (*it)->canBeAbortedAfterCertainTime()){
//					std::cout << "Abort training, has worked: " << (*it)->getWorkedAmountOfSeconds() << std::endl;
					(*it)->abortTraing(); // break the training
				}
				if(selectedValue != m_waitingList.end() && !selectedValueWasUsed){
					if(!(*it)->shouldTrainingBePaused() && !(*it)->isWaiting() && (*it)->runningTimeSinceLastWait() > 2.0){ // only change thread if it is running more than 2 seconds
						if((int) bestAttractionLevel > (int) (*it)->calcAttractionLevel(minAmountOfPoints, maxAmountOfPoints) || (*selectedValue)->getPriority() > (*it)->getPriority()){ // must be at least 1. point be better
							// hold this training and start the other one
							(*it)->pauseTheTraining(); // pause the actual training
							selectedValueWasUsed = true;
						}
					}
				}
				if((*it)->isWaiting()){
					// there is a running thread which waits -> put him back in the waiting list
//					std::cout << "A thread was moved from waiting to paused!" << std::endl;
					PackageList::const_iterator copyIt = it;
					m_waitingList.push_back(*it); // append at the waiting list
					--it; // go one back, in the end of the loop the next element will be taken
					m_runningList.erase(copyIt); // erase the copied element
					--m_counter; // -> now a new thread can run
					continue; // without continue the first element could be made to the "zero" element, which does not exists -> seg fault
				}
				if((*it)->isTaskFinished()){
//					std::cout << "A thread is finished!" << std::endl;
					PackageList::const_iterator copyIt = it; // perform copy
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
		usleep(m_timeToSleep * 1e6);
	}
}

bool ThreadMaster::appendThreadToList(InformationPackage* package){
	bool ret = false;
	m_mutex.lock();
	m_waitingList.push_back(package);
	ret = true;
	m_mutex.unlock();
	return ret;
}

void ThreadMaster::abortAllThreads(){
	m_mutex.lock();
	for(PackageList::const_iterator it = m_waitingList.begin(); it != m_waitingList.end(); ++it){
		if((*it)->canBeAbortedInGeneral()){
			(*it)->abortTraing();
		}
	}
	for(PackageList::const_iterator it = m_runningList.begin(); it != m_runningList.end(); ++it){
		if((*it)->canBeAbortedInGeneral()){
			(*it)->abortTraing();
		}
	}
	m_mutex.unlock();
}
