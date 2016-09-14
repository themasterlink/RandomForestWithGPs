/*
 * StopWatch.cc
 *
 *  Created on: 01.06.2016
 *      Author: Max
 */

#include "StopWatch.h"

StopWatch::StopWatch(){
	m_start = m_stop = boost::posix_time::microsec_clock::local_time();
	counter = 1;
	avgTime = 0.0;
}

StopWatch::~StopWatch(){
}

void StopWatch::recordActTime(){
	const double time = elapsedSeconds();
	const double fac = 1.0 / counter;
	avgTime = fac * time + (1-fac) * avgTime;
	startTime();
	++counter;
}

const std::string StopWatch::elapsedAvgAsPrettyTime() const{
	std::stringstream ss;
	ss << TimeFrame(avgTime);
	return ss.str();
}


