/*
 * ThreadMaster.cc
 *
 *  Created on: 22.11.2016
 *      Author: Max
 */

#include "ThreadMaster.h"


InformationPackage::InformationPackage(InfoType type,
		double correctlyClassified,
		int amountOfPoints): m_type(type),
		m_performTask(false),
		m_correctlyClassified(correctlyClassified),
		m_amountOfAffectedPoints(amountOfPoints),
		m_amountOfTrainingsSteps(0) {
};



int ThreadMaster::m_counter = 0;
int ThreadMaster::m_maxCounter = 0;
double ThreadMaster::m_timeToSleep = 1.0;
ThreadMaster::PackageList ThreadMaster::m_packages;
boost::thread* ThreadMaster::m_mainThread = nullptr;
boost::mutex ThreadMaster::m_mutex;


ThreadMaster::ThreadMaster() {
	// TODO Auto-generated constructor stub

}

ThreadMaster::~ThreadMaster() {
	// TODO Auto-generated destructor stub
}


void ThreadMaster::run(){
	int nrOfInducingPoints;
	Settings::getValue("IVM.nrOfInducingPoints", nrOfInducingPoints);
	const int amountOfPointsNeededForIvms = nrOfInducingPoints * 1.2;
	while(true){
		m_mutex.lock();
		if(m_counter < m_maxCounter){
			PackageList::const_iterator selectedValue = m_packages.end();
			double attractionLevel = 0;
			int minAmountOfPoints, maxAmountOfPoints;
			for(PackageList::const_iterator it = m_packages.begin(); it != m_packages.end(); ++it){
				const int amount = (*it)->amountOfAffectedPoints();
				if(minAmountOfPoints > amount){
					minAmountOfPoints = amount;
				}
				if(maxAmountOfPoints < amount){
					maxAmountOfPoints = amount;
				}
			}
			for(PackageList::const_iterator it = m_packages.begin(); it != m_packages.end(); ++it){
				const int amount = (*it)->amountOfAffectedPoints();
				const double correct = (*it)->correctlyClassified();
				switch((*it)->getType()){
				case InfoType::IVM_TRAIN:
					if(selectedValue == m_packages.end()){
						if(amount > amountOfPointsNeededForIvms){
							selectedValue = it;
						}
					}else{
						double partAmount = 1.;
						if(maxAmountOfPoints != minAmountOfPoints){
							partAmount = (((double) amount - minAmountOfPoints)
									/ (double)(maxAmountOfPoints - minAmountOfPoints)) * 100.;
						}
						const double actAttractionLevel = partAmount * correct;

					}
					break;
				case InfoType::ORF_TRAIN:
					break;
				default:
					printError("This type is not supported here!");
					break;
				}
			}
		}
		m_mutex.unlock();
		usleep(m_timeToSleep * 1e6);
	}
}

bool ThreadMaster::appendThreadToList(InformationPackage* package){
	bool ret = false;
	m_mutex.lock();
	m_packages.push_back(package);
	ret = true;
	m_mutex.unlock();
	return ret;
}
