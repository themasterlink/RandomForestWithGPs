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
#include "../Utility/StopWatch.h"
#include "InformationPackage.h"
#include "Types.h"
#include <atomic>

class ThreadMaster {
	friend ScreenOutput;

SingeltonMacro(ThreadMaster);

public:

	using InfoType = InformationPackage::InfoType;

	void start();

	void setFrequence(const Real frequence);

	bool appendThreadToList(InformationPackage* package);

	void threadHasFinished(InformationPackage* package);

	void setMaxCounter();

	void abortAllThreads();

	const unsigned int getAmountOfThreads();

	void stopExecution(){ m_keepRunning = false; }

	void blockUntilFinished(){
		m_isFinished.lock();
		m_isFinished.unlock();
	};

private:
	void run();

	void sortWaitingList(const int minAmountOfPoints, const int maxAmountOfPoints);

	using PackageList = std::list<InformationPackage*>;
	using PackageListIterator = PackageList::iterator;
	using PackageListConstIterator = PackageList::const_iterator;

	PackageList m_waitingList;

	PackageList m_runningList;

	int m_counter;

	Real m_timeToSleep;

	boost::thread* m_mainThread;

	Mutex m_mutex;

	unsigned int m_maxCounter;

	std::atomic<bool> m_keepRunning;

	Mutex m_isFinished;

};



#endif /* BASE_THREADMASTER_H_ */
