/*
 * ThreadSafeCounter.h
 *
 *  Created on: 23.07.2016
 *      Author: Max
 */

#ifndef UTILITY_THREADSAFETHREADCOUNTER_H_
#define UTILITY_THREADSAFETHREADCOUNTER_H_

#include "../Base/Types.h"

class ThreadSafeThreadCounter {
public:
	ThreadSafeThreadCounter();
	virtual ~ThreadSafeThreadCounter();
	// returns true if adding is possible
	bool addNewThread();

	// decrements the counter
	void removeThread();
	
	// call only if you are sure, the number of threads is below the number of possible threads!
	void forceAddThread();

	// unsafe
	int currentThreadCount(){ return m_counter; };

private:
	const unsigned int m_maxNr;
	int m_counter;
	Mutex m_mutex;
};

#endif /* UTILITY_THREADSAFETHREADCOUNTER_H_ */
