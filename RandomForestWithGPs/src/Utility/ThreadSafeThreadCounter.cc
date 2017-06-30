/*
 * ThreadSafeCounter.cc
 *
 *  Created on: 23.07.2016
 *      Author: Max
 */

#include "ThreadSafeThreadCounter.h"
#include "../Base/ThreadMaster.h"
#include "Util.h"

ThreadSafeThreadCounter::ThreadSafeThreadCounter():
		m_maxNr(ThreadMaster::instance().getAmountOfThreads()),
		m_counter(0) {
}

ThreadSafeThreadCounter::~ThreadSafeThreadCounter() {

}

// returns true if adding is possible
bool ThreadSafeThreadCounter::addNewThread(){
	m_mutex.lock();
	if(m_counter < m_maxNr){
		++m_counter;
		m_mutex.unlock();
		return true;
	}
	m_mutex.unlock();
	return false;
}

// decrements the counter
void ThreadSafeThreadCounter::removeThread(){
	lockStatementWith(--m_counter, m_mutex);
}

// call only if you are sure, the number of threads is below the number of possible threads!
void ThreadSafeThreadCounter::forceAddThread(){
	lockStatementWith(++m_counter, m_mutex);
}
