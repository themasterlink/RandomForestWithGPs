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

SingeltonMacro(GlobalStopWatch);

public:

	void startTime(){
		lockStatementWith(m_sw.startTime(), m_mutex);
	}

	Real recordActTime(){
		lockStatementWithSave(m_sw.recordActTime(), Real var, m_mutex);
		return var;
	}

	TimeFrame elapsedAvgAsTimeFrame(){
		lockStatementWithSave(m_sw.elapsedAvgAsTimeFrame(), TimeFrame var, m_mutex);
		return var;
	}

private:

	Mutex m_mutex;

	StopWatch m_sw;

};

template<typename currentWatch>
GlobalStopWatch<currentWatch>::GlobalStopWatch(){};


#endif //RANDOMFORESTWITHGPS_GLOBALSTOPWATCH_H
