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

	int amountOfTrainingStepsPerformed() const { return m_amountOfTrainingsSteps; };

	void finishedTask(){ m_performTask = true; };

	bool isTaskFinished(){ return m_performTask; };

	bool shouldTrainingBeAborted(){ return m_abortTraining; };

	void abortTraing(){ m_abortTraining = true; };

	void wait(){
		boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(m_mutex);
		m_condition.wait(lock);
	};

	void notify(){ m_condition.notify_all(); m_sw.startTime(); };

	InfoType getType(){ return m_type; };

	StopWatch& getWatch(){return m_sw;};

private:

	const InfoType m_type;

	boost::interprocess::interprocess_condition m_condition;

	boost::interprocess::interprocess_mutex m_mutex;

	bool m_performTask;

	bool m_abortTraining;

	double m_correctlyClassified;

	int m_amountOfAffectedPoints;

	int m_amountOfTrainingsSteps;

	StopWatch m_sw;

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

	static PackageList m_waitingList;

	static PackageList m_runningList;

	static int m_counter;

	static double m_timeToSleep;

	static boost::thread* m_mainThread;

	static boost::mutex m_mutex;

	static int m_maxCounter;

};



#endif /* BASE_THREADMASTER_H_ */
