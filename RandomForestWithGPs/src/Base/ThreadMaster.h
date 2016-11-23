/*
 * ThreadMaster.h
 *
 *  Created on: 22.11.2016
 *      Author: Max
 */

#ifndef BASE_THREADMASTER_H_
#define BASE_THREADMASTER_H_

#include <boost/thread.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include "../Utility/Util.h"
#include "Settings.h"

class InformationPackage {
public:
	enum InfoType {
		ORF_TRAIN = 0,
		IVM_TRAIN = 1
	};

	InformationPackage(InfoType type, double correctlyClassified, int amountOfPoints);

	double correctlyClassified() const { return m_correctlyClassified; };

	int amountOfAffectedPoints() const { return m_amountOfAffectedPoints; };

	void changeCorrectlyClassified(const double newValue){ m_correctlyClassified = newValue; };

	void changeAmountOfAffectedPoints(const int newAmount){ m_amountOfAffectedPoints = newAmount; };

	void performedOneTrainingStep(){ ++m_amountOfTrainingsSteps; };

	void setPerformTaskFlag(bool doTask){ m_performTask = doTask; };

	void wait(){
		boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(m_mutex);
		m_condition.wait(lock);
	};

	void notify(){ m_condition.notify_all(); };

	InfoType getType(){ return m_type; };

private:

	const InfoType m_type;

	boost::interprocess::interprocess_condition m_condition;

	boost::interprocess::interprocess_mutex m_mutex;

	bool m_performTask;

	double m_correctlyClassified;

	int m_amountOfAffectedPoints;

	int m_amountOfTrainingsSteps;

};

class ThreadMaster {
public:

	typedef InformationPackage::InfoType InfoType;

	static void start();

	static void setFrequence(const double frequence);

	static bool appendThreadToList(InformationPackage* package);

	static void threadHasFinished(InformationPackage* package);

	static void setMaxCounter(){ m_maxCounter = boost::thread::hardware_concurrency();}

private:
	static void run();

	ThreadMaster();
	virtual ~ThreadMaster();

	typedef std::list<InformationPackage*> PackageList;

	static PackageList m_packages;

	static int m_counter;

	static double m_timeToSleep;

	static boost::thread* m_mainThread;

	static boost::mutex m_mutex;

	static int m_maxCounter;

};

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


#endif /* BASE_THREADMASTER_H_ */
