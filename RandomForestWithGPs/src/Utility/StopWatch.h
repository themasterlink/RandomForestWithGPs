/*
 * StopWatch.h
 *
 *  Created on: 01.06.2016
 *      Author: Max
 */

#ifndef UTILITY_STOPWATCH_H_
#define UTILITY_STOPWATCH_H_

#include "boost/date_time/posix_time/posix_time.hpp"
#include "TimeFrame.h"
#include "AvgNumber.h"

class StopWatch {
public:
	StopWatch();

	virtual ~StopWatch();

	void startTime();

	void stopTime();

	Real elapsedSeconds() const;

	Real elapsedMiliSeconds() const;

	TimeFrame elapsedAsTimeFrame() const;

	TimeFrame elapsedAvgAsTimeFrame() const;

	const std::string elapsedAsPrettyTime() const;

	Real recordActTime();

	const std::string elapsedAvgAsPrettyTime() const;

	static Real getActTime();

	void addNewAvgTime(const Real seconds);

	// returns the counter from the avg time measurment
	unsigned int getAvgCounter() const{ return (unsigned int) avgTime.counter(); };

private:

	using Time = boost::posix_time::ptime;
	using TimeDuration = boost::posix_time::time_duration;

	Time m_start, m_stop;
	AvgNumber avgTime;
};

inline
void StopWatch::startTime(){
	m_start = boost::posix_time::microsec_clock::local_time();
}

inline
void StopWatch::stopTime(){
	m_stop = boost::posix_time::microsec_clock::local_time();
}

inline
Real StopWatch::elapsedSeconds() const{
	return (Real) ((boost::posix_time::microsec_clock::local_time() - m_start).total_milliseconds() / 1000.0);
}

inline
Real StopWatch::elapsedMiliSeconds() const{
	return (boost::posix_time::microsec_clock::local_time() - m_start).total_milliseconds();
}

inline
TimeFrame StopWatch::elapsedAsTimeFrame() const{
	return TimeFrame(elapsedSeconds());
}

inline
const std::string StopWatch::elapsedAsPrettyTime() const{
	std::stringstream ss;
	ss << TimeFrame(elapsedSeconds());
	return ss.str();
}

inline
TimeFrame StopWatch::elapsedAvgAsTimeFrame() const{
	return TimeFrame(avgTime.mean());
}

inline
Real StopWatch::getActTime(){
	static Time startOfTime(boost::gregorian::date(1970, 1, 1));
	return (Real) ((boost::posix_time::microsec_clock::local_time() - startOfTime).total_milliseconds() / 1000.0);
}

inline
void StopWatch::addNewAvgTime(const Real seconds){
	avgTime.addNew(seconds);
}
#endif /* UTILITY_STOPWATCH_H_ */
