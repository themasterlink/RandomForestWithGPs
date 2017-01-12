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

class StopWatch{
public:
	StopWatch();
	virtual ~StopWatch();

	void startTime();
	void stopTime();
	double elapsedSeconds() const;
	double elapsedMiliSeconds() const;

	TimeFrame elapsedAsTimeFrame() const;

	TimeFrame elapsedAvgAsTimeFrame() const;

	const std::string elapsedAsPrettyTime() const;

	void recordActTime();

	const std::string elapsedAvgAsPrettyTime() const;

	static double getActTime();

	// returns the counter from the avg time measurment
	unsigned int getAvgCounter() const { return counter; };

private:

	typedef boost::posix_time::ptime Time;
	typedef boost::posix_time::time_duration TimeDuration;

	Time m_start, m_stop;
	int counter;
	double avgTime;
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
double StopWatch::elapsedSeconds() const{
	return (boost::posix_time::microsec_clock::local_time() - m_start).total_milliseconds() / 1000.0;
}

inline
double StopWatch::elapsedMiliSeconds() const{
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
	return TimeFrame(avgTime);
}

inline
double StopWatch::getActTime(){
	static Time startOfTime(boost::gregorian::date(1970,1,1));
	return (boost::posix_time::microsec_clock::local_time() - startOfTime).total_milliseconds() / 1000.0;
}

#endif /* UTILITY_STOPWATCH_H_ */
