/*
 * TreeCounter.h
 *
 *  Created on: 26.10.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_TREECOUNTER_H_
#define RANDOMFORESTS_TREECOUNTER_H_

#include "../Utility/Util.h"

#ifdef BUILD_OLD_CODE

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

#endif // BUILD_OLD_CODE

#endif /* RANDOMFORESTS_TREECOUNTER_H_ */
