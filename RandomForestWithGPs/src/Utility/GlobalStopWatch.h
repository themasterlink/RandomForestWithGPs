//
// Created by denn_ma on 6/30/17.
//

#ifndef RANDOMFORESTWITHGPS_GLOBALSTOPWATCH_H
#define RANDOMFORESTWITHGPS_GLOBALSTOPWATCH_H

#include "StopWatch.h"

struct DynamicDecisionTreeTrain {
};
struct BigDecisionTreeTrain {
};


template<typename currentWatch>
class GlobalStopWatch {

SINGELTON_MACRO(GlobalStopWatch);

public:

	void recordActTime(const Real seconds){
		lockStatementWith(m_sw.addNewAvgTime(seconds), m_mutex);
	}

	TimeFrame elapsedAvgAsTimeFrame(){
		lockStatementWithSave(m_sw.elapsedAvgAsTimeFrame(), TimeFrame var, m_mutex);
		return var;
	}

	unsigned long getAvgCounter(){
		lockStatementWithSave(m_sw.getAvgCounter(), const unsigned long var, m_mutex);
		return var;
	}

private:

	Mutex m_mutex;

	StopWatch m_sw;

};

template<typename currentWatch>
GlobalStopWatch<currentWatch>::GlobalStopWatch(){};


#endif //RANDOMFORESTWITHGPS_GLOBALSTOPWATCH_H
