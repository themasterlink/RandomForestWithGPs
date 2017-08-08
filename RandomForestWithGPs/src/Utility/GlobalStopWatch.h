//
// Created by denn_ma on 6/30/17.
//

#ifndef RANDOMFORESTWITHGPS_GLOBALSTOPWATCH_H
#define RANDOMFORESTWITHGPS_GLOBALSTOPWATCH_H

#include "StopWatch.h"
#include "../Base/Singleton.h"

struct DynamicDecisionTreeTrain;
struct BigDecisionTreeTrain;
struct DDDTAvgSizeOfUsedStorage;


template<typename currentCounter>
class GlobalCounter : public Singleton<GlobalCounter<currentCounter> > {

	friend Singleton<GlobalCounter< currentCounter> >;

public:

	void addNew(Real val){
		lockStatementWith(m_nr.addNew(val), m_mutex);
	}

	Real mean(){
		lockStatementWithSave(m_nr.mean(), const auto val, m_mutex);
		return val;
	}

	unsigned long counter(){
		lockStatementWithSave(m_nr.counter(), const auto val, m_mutex);
		return val;
	}

	void reset(){
		lockStatementWith(m_nr.reset(), m_mutex);
	}

private:
	GlobalCounter() = default;
	~GlobalCounter() = default;

	AvgNumber m_nr;

	Mutex m_mutex;

};

template<typename currentWatch>
class GlobalStopWatch  : public Singleton<GlobalStopWatch<currentWatch> > {

	friend Singleton<GlobalStopWatch<currentWatch> >;

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

	GlobalStopWatch() = default;
	~GlobalStopWatch() = default;

};

#endif //RANDOMFORESTWITHGPS_GLOBALSTOPWATCH_H
