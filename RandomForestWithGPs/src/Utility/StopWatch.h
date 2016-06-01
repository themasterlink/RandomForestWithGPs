/*
 * StopWatch.h
 *
 *  Created on: 01.06.2016
 *      Author: Max
 */

#ifndef UTILITY_STOPWATCH_H_
#define UTILITY_STOPWATCH_H_

#include "boost/date_time/posix_time/posix_time.hpp"


class StopWatch {
public:
	StopWatch();
	virtual ~StopWatch();

	void startTime();
	void stopTime();
	double elapsedSeconds();
	double elapsedMiliSeconds();

private:

	typedef boost::posix_time::ptime Time;
	typedef boost::posix_time::time_duration TimeDuration;


	Time m_start, m_stop;
};

#endif /* UTILITY_STOPWATCH_H_ */
