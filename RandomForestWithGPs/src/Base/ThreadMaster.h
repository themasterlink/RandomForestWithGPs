/*
 * ThreadMaster.h
 *
 *  Created on: 22.11.2016
 *      Author: Max
 */

#ifndef BASE_THREADMASTER_H_
#define BASE_THREADMASTER_H_

#include "../Utility/StopWatch.h"
#include "InformationPackage.h"
#include "Types.h"
#include <atomic>
#include "Singleton.h"
#include "Thread.h"

class ScreenOutput;

class ThreadMaster : public Singleton<ThreadMaster> {

	friend ScreenOutput;
	friend Singleton<ThreadMaster>;

public:

	using InfoType = InformationPackage::InfoType;

	void start();

	void setFrequence(const Real frequence);

	bool appendThreadToList(InformationPackage* package);

	void threadHasFinished(SharedPtr<InformationPackage>&& package);

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

	UniquePtr<Thread> m_mainThread;

	Mutex m_mutex;

	unsigned int m_maxCounter;

	std::atomic<bool> m_keepRunning;

	Mutex m_isFinished;

	ThreadMaster();
	~ThreadMaster() = default;

};



#endif /* BASE_THREADMASTER_H_ */
