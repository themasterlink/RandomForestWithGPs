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
public:

	using InfoType = InformationPackage::InfoType;

	static void start();

	static void setFrequence(const Real frequence);

	static bool appendThreadToList(InformationPackage* package);

	static void threadHasFinished(InformationPackage* package);

	static void setMaxCounter();

	static void abortAllThreads();

	static const unsigned int getAmountOfThreads();

	static void stopExecution(){m_keepRunning = false;}

	static void blockUntilFinished(){m_isFinished.lock(); m_isFinished.unlock();};

private:
	static void run();

	static void sortWaitingList(const int minAmountOfPoints, const int maxAmountOfPoints);

	ThreadMaster() = default;
	~ThreadMaster() = default;

	using PackageList = std::list<InformationPackage*>;
	using PackageListIterator = PackageList::iterator;
	using PackageListConstIterator = PackageList::const_iterator;

	static PackageList m_waitingList;

	static PackageList m_runningList;

	static int m_counter;

	static Real m_timeToSleep;

	static boost::thread* m_mainThread;

	static Mutex m_mutex;

	static unsigned int m_maxCounter;

	static std::atomic<bool> m_keepRunning;

	static Mutex m_isFinished;

};



#endif /* BASE_THREADMASTER_H_ */
