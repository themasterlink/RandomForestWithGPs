/*
 * TreeCounter.h
 *
 *  Created on: 26.10.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_TREECOUNTER_H_
#define RANDOMFORESTS_TREECOUNTER_H_

#include <boost/thread.hpp> // Boost threads

class TreeCounter{
public:
	TreeCounter() : counter(0){};

	void addToCounter(const int val){
		mutex.lock();
		counter += val;
		mutex.unlock();
	}

	void addOneToCounter(){
		mutex.lock();
		++counter;
		mutex.unlock();
	}

	int getCounter() const{
		return counter;
	}
private:
	Mutex mutex;
	int counter;
};


#endif /* RANDOMFORESTS_TREECOUNTER_H_ */
