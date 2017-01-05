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

class ThreadMaster {
	friend ScreenOutput;
public:

	typedef InformationPackage::InfoType InfoType;

	static void start();

	static void setFrequence(const double frequence);

	static bool appendThreadToList(InformationPackage* package);

	static void threadHasFinished(InformationPackage* package);

	static void setMaxCounter(){ m_maxCounter = 1; } //boost::thread::hardware_concurrency();}

	static void abortAllThreads();

private:
	static void run();

	static void sortWaitingList(const int amountOfPointsNeededForIvms, const int minAmountOfPoints, const int maxAmountOfPoints);

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
