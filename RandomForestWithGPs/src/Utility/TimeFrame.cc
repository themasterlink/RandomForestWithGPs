/*
 * TimeFrame.cc
 *
 *  Created on: 07.06.2016
 *      Author: Max
 */

#include "TimeFrame.h"

const TimeFrame operator*(const TimeFrame& frame, const double fac){
	TimeFrame newFrame(frame);
	newFrame *= fac;
	return newFrame;
}

const TimeFrame operator-(const TimeFrame& lhs, const TimeFrame& rhs){
	TimeFrame newFrame(lhs);
	newFrame -= rhs;
	return newFrame;
}

const TimeFrame operator+(const TimeFrame& lhs, const TimeFrame& rhs){
	TimeFrame newFrame(lhs);
	newFrame += rhs;
	return newFrame;
}

TimeFrame::TimeFrame(const double seconds){
	setWithSeconds(seconds);
}

void TimeFrame::setWithSeconds(const double seconds){
	if(seconds > 3600){
		this->m_hours = (int) seconds / 3600;
		this->m_minutes = (int) (seconds - this->m_hours * 3600) / 60;
		this->m_seconds = fmod(seconds, 60.0);
	}else if(seconds > 60){
		this->m_hours = 0;
		this->m_minutes = (int) seconds / 60;
		this->m_seconds = fmod(seconds, 60.0);
	}else{
		this->m_hours = 0;
		this->m_minutes = 0;
		this->m_seconds = seconds;
	}
}

std::ostream& operator<<(std::ostream& stream, const TimeFrame& time){
	if(time.m_hours > 0 && time.m_minutes > 0){
		char times[40];
		sprintf(times, "%d h, %d min, %2.2f sec", time.m_hours, time.m_minutes, time.m_seconds);
		stream << times;
		return stream;
	}else if(time.m_minutes > 0){
		char times[40];
		sprintf(times, "%d min, %2.2f sec", time.m_minutes, time.m_seconds);
		stream << times;
		return stream;
	}else{
		char seconds[14];
		sprintf(seconds, "%2.2f sec", time.m_seconds);
		stream << seconds;
		return stream;
	}
}

TimeFrame& TimeFrame::operator*=(const double fac){
	setWithSeconds(getSeconds() * fac);
	return *this;
}

TimeFrame& TimeFrame::operator+=(const TimeFrame& rhs){
	setWithSeconds(getSeconds() + rhs.getSeconds());
	return *this;
}

TimeFrame& TimeFrame::operator-=(const TimeFrame& rhs){
	setWithSeconds(getSeconds() - rhs.getSeconds());
	return *this;
}
