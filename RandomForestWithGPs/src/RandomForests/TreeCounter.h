/*
 * TreeCounter.h
 *
 *  Created on: 26.10.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_TREECOUNTER_H_
#define RANDOMFORESTS_TREECOUNTER_H_

#include "../Utility/Util.h"

class TreeCounter{
public:
	TreeCounter(): m_counter(0){};

	void addToCounter(const int val){
		lockStatementWith(m_counter += val, m_mutex);
	}

	void addOneToCounter(){
		lockStatementWith(++m_counter, m_mutex);
	}

	int getCounter() const{
		return m_counter;
	}
private:
	Mutex m_mutex;
	int m_counter;
};


#endif /* RANDOMFORESTS_TREECOUNTER_H_ */
