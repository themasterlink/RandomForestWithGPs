/*
 * StopWatch.cc
 *
 *  Created on: 01.06.2016
 *      Author: Max
 */

#include "StopWatch.h"

StopWatch::StopWatch() {
	m_start = m_stop =  boost::posix_time::microsec_clock::local_time();
}

StopWatch::~StopWatch() {
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

