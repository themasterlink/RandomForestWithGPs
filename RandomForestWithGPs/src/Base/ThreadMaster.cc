/*
 * ThreadMaster.cc
 *
 *  Created on: 22.11.2016
 *      Author: Max
 */

#include "ThreadMaster.h"
#include "../Utility/Util.h"
#include "Logger.h"
#include "Settings.h"
#include "CommandSettings.h"

InformationPackage::InformationPackage(InfoType type,
		double correctlyClassified,
		int amountOfPoints): m_type(type),
		m_performTask(false),
		m_abortTraining(false),
		m_isWaiting(false),
		m_shouldTrainingBeHold(false),
		m_correctlyClassified(correctlyClassified),
		m_amountOfAffectedPoints(amountOfPoints),
		m_amountOfTrainingsSteps(0),
		m_workedTime(0){
};


double InformationPackage::calcAttractionLevel(const int minAmountOfPoints, const int maxAmountOfPoints){
	double partAmount = 100.; // if all have the same size, just ignore this value and go after the correct classified amount
	if(maxAmountOfPoints != minAmountOfPoints){
		partAmount = (((double) m_amountOfAffectedPoints - minAmountOfPoints)
				/ (double)(maxAmountOfPoints - minAmountOfPoints)) * 100.;
	}
	return partAmount + (100 - m_correctlyClassified);
}

void InformationPackage::wait(){
	if(!m_abortTraining){ // only wait if the training is not aborted
		m_shouldTrainingBeHold = false; // could be called, because the training should be hold
		m_isWaiting = true;
		m_workedTime += m_sw.elapsedSeconds();
		boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(m_mutex);
		m_condition.wait(lock);
	}
};

void InformationPackage::notify(){
	m_condition.notify_all();
	m_sw.startTime();
	m_isWaiting = false;
};

double InformationPackage::getWorkedAmountOfSeconds(){
	if(!m_isWaiting){
		return m_sw.elapsedSeconds() + m_workedTime;
	}
	return m_workedTime; // if it is waiting at the moment
}

void InformationPackage::setAdditionalInfo(const std::string& line){
	m_lineMutex.lock();
	m_additionalInformation = line;
	m_lineMutex.unlock();
}

void InformationPackage::printLineToScreenForThisThread(const std::string& line) {
	m_lineMutex.lock();
	m_lines.push_back(line);
	Logger::addToFile(m_standartInfo+"\n\t"+line);
	m_lineMutex.unlock();
}

void InformationPackage::overwriteLastLineToScreenForThisThread(const std::string& line){
	if(m_lines.size() > 0){
		m_lineMutex.lock();
		m_lines.back() = line;
		Logger::addToFile(m_standartInfo+"\n\toverwrite: "+line);
		m_lineMutex.unlock();
	}
}

bool InformationPackage::canBeAborted(){
	switch(m_type){
	case ORF_TRAIN:
		return true;
	case IVM_TRAIN:
		return true;
	case IVM_PREDICT:
		return false;
	default:
		printError("This type is unknown!");
		return false;
	}
}


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
		int minAmountOfPoints, maxAmountOfPoints;
		for(PackageList::const_iterator it = m_waitingList.begin(); it != m_waitingList.end(); ++it){
			const int amount = (*it)->amountOfAffectedPoints();
			if(minAmountOfPoints > amount){
				minAmountOfPoints = amount;
			}
			if(maxAmountOfPoints < amount){
				maxAmountOfPoints = amount;
			}
		}
		for(PackageList::const_iterator it = m_waitingList.begin(); it != m_waitingList.end(); ++it){
			if(!(*it)->isWaiting()){
				continue; // hasn't reached the waiting point so the thread should be added to the running list
			}
			const int amount = (*it)->amountOfAffectedPoints();
			const double correct = (*it)->correctlyClassified();
			switch((*it)->getType()){
			case InfoType::IVM_TRAIN:
				if(selectedValue == m_waitingList.end()){
					if(amount > amountOfPointsNeededForIvms){
						selectedValue = it;
					}
				}else{
					const double attractionLevel = (*it)->calcAttractionLevel(minAmountOfPoints, maxAmountOfPoints);
					if(attractionLevel > bestAttractionLevel){
						bestAttractionLevel = attractionLevel;
						selectedValue = it;
					}
				}
				break;
			case InfoType::ORF_TRAIN:
				if(selectedValue == m_waitingList.end()){
					selectedValue = it;
				}else{
					const double attractionLevel = (*it)->calcAttractionLevel(minAmountOfPoints, maxAmountOfPoints);
					if(attractionLevel > bestAttractionLevel){
						bestAttractionLevel = attractionLevel;
						selectedValue = it;
					}
				}
				break;
			case InfoType::IVM_PREDICT:
				if(selectedValue == m_waitingList.end()){
					selectedValue = it;
				}else{
					const int diff = (*selectedValue)->amountOfTrainingStepsPerformed() - (*it)->amountOfTrainingStepsPerformed();
					const int amountOfPoints = ((*it)->amountOfAffectedPoints() + (*selectedValue)->amountOfAffectedPoints()) * 0.5;
					if(diff > 0.1 * amountOfPoints){
						selectedValue = it;
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
				(*selectedValue)->notify(); // start running of the thread
				m_waitingList.erase(selectedValue); // remove the selected value out of the waiting thread list
				selectedValue = m_waitingList.end();
			}
		}
		for(PackageList::const_iterator it = m_runningList.begin(); it != m_runningList.end(); ++it){
			// for each running element check if execution is finished
			if((*it)->getWorkedAmountOfSeconds() > 5.0 || (*it)->isTaskFinished()){ // each training have to take at least 5 seconds!
				if((*it)->getWorkedAmountOfSeconds() > CommandSettings::get_samplingAndTraining() && !(*it)->shouldTrainingBeAborted() && (*it)->canBeAborted()){
//					std::cout << "Abort training, has worked: " << (*it)->getWorkedAmountOfSeconds() << std::endl;
					(*it)->abortTraing(); // break the training
				}
				if(selectedValue != m_waitingList.end() && !selectedValueWasUsed){
					if(!(*it)->shouldTrainingBePaused()){
						if(bestAttractionLevel > (*it)->calcAttractionLevel(minAmountOfPoints, maxAmountOfPoints)){
							// hold this training and start the other one
							(*it)->pauseTheTraining(); // pause the actual training
							selectedValueWasUsed = true;
//							std::cout << "A thread should wait!" << std::endl;
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
