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

void StopWatch::startTime(){
	m_start = boost::posix_time::microsec_clock::local_time();
}
void StopWatch::stopTime(){
	m_stop = boost::posix_time::microsec_clock::local_time();
}

double StopWatch::elapsedSeconds(){
	return (boost::posix_time::microsec_clock::local_time() - m_start).total_milliseconds() / 1000.0;
}

double StopWatch::elapsedMiliSeconds(){
	return (boost::posix_time::microsec_clock::local_time() - m_start).total_milliseconds();
}

TimeFrame StopWatch::elapsedAsTimeFrame(){
	return TimeFrame(elapsedSeconds());
}

std::string StopWatch::elapsedAsPrettyTime(){
	std::stringstream ss;
	ss << TimeFrame(elapsedSeconds());
	return ss.str();
}

void StopWatch::recordActTime(){
	const double time = elapsedSeconds();
	const double fac = 1.0 / counter;
	avgTime = fac * time + (1-fac) * avgTime;
	startTime();
	++counter;
}

std::string StopWatch::elapsedAvgAsPrettyTime(){
	std::stringstream ss;
	ss << TimeFrame(avgTime);
	return ss.str();
}


