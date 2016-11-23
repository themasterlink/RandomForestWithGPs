/*
 * ThreadMaster.cc
 *
 *  Created on: 22.11.2016
 *      Author: Max
 */

#include "ThreadMaster.h"
#include "CommandSettings.h"

InformationPackage::InformationPackage(InfoType type,
		double correctlyClassified,
		int amountOfPoints): m_type(type),
		m_performTask(false),
		m_abortTraining(false),
		m_correctlyClassified(correctlyClassified),
		m_amountOfAffectedPoints(amountOfPoints),
		m_amountOfTrainingsSteps(0) {
};



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

}

void ThreadMaster::run(){
	int nrOfInducingPoints;
	Settings::getValue("IVM.nrOfInducingPoints", nrOfInducingPoints);
	const int amountOfPointsNeededForIvms = nrOfInducingPoints * 1.2;
	while(true){
		m_mutex.lock();
		if(m_counter < m_maxCounter){
			PackageList::const_iterator selectedValue = m_waitingList.end();
			double attractionLevel = 0;
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
				const int amount = (*it)->amountOfAffectedPoints();
				const double correct = (*it)->correctlyClassified();
				switch((*it)->getType()){
				case InfoType::IVM_TRAIN:
					if(selectedValue == m_waitingList.end()){
						if(amount > amountOfPointsNeededForIvms){
							selectedValue = it;
						}
					}else{
						double partAmount = 1.;
						if(maxAmountOfPoints != minAmountOfPoints){
							partAmount = (((double) amount - minAmountOfPoints)
									/ (double)(maxAmountOfPoints - minAmountOfPoints)) * 100.;
						}
						const double actAttractionLevel = partAmount + (100 - correct);

					}
					break;
				case InfoType::ORF_TRAIN:
					break;
				default:
					printError("This type is not supported here!");
					break;
				}
			}
			if(selectedValue != m_waitingList.end()){
				m_runningList.push_back(*selectedValue); // first add to the running list
				++m_counter; // increase the counter of running threads
				(*selectedValue)->notify(); // start running of the thread
				m_waitingList.erase(selectedValue); // remove the selected value out of the waiting thread list
			}
		}
		for(PackageList::const_iterator it = m_runningList.begin(); it != m_runningList.end(); ++it){
			// for each running element check if execution is finished
			if((*it)->getWatch().elapsedSeconds() > 5.0){ // each training have to take at least 5 seconds!
				if((*it)->getWatch().elapsedSeconds() > CommandSettings::get_samplingAndTraining()){
					(*it)->abortTraing(); // break the training
				}
				if((*it)->isTaskFinished()){
					PackageList::const_iterator copyIt = it; // perform copy
					--it; // go one back, in the end of the loop the next element will be taken
					m_runningList.erase(copyIt); // erase the copied element
					// decrease the counter
					--m_counter; // -> now a new thread can run
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
