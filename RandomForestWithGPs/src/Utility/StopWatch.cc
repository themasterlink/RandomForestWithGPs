/*
 * StopWatch.cc
 *
 *  Created on: 01.06.2016
 *      Author: Max
 */

#include "StopWatch.h"

StopWatch::StopWatch(){
	m_start = m_stop = boost::posix_time::microsec_clock::local_time();
}

StopWatch::~StopWatch(){
}

void StopWatch::recordActTime(){
	avgTime.addNew(elapsedSeconds());
	startTime();
}

const std::string StopWatch::elapsedAvgAsPrettyTime() const{
	std::stringstream ss;
	ss << TimeFrame(avgTime.mean());
	return ss.str();
}


