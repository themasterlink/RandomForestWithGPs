/*
 * TimeFrame.cc
 *
 *  Created on: 07.06.2016
 *      Author: Max
 */

#include "TimeFrame.h"
#include "StringHelper.h"

const TimeFrame operator*(const TimeFrame& frame, const Real fac){
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

TimeFrame::TimeFrame(const Real seconds){
	setWithSeconds(seconds);
}

void TimeFrame::setWithSeconds(const Real seconds){
	if(seconds > 3600){
		this->m_hours = (int) seconds / 3600;
		this->m_minutes = (int) (seconds - this->m_hours * 3600) / 60;
		this->m_seconds = modReal(seconds, (Real) 60.0);
	}else if(seconds > 60){
		this->m_hours = 0;
		this->m_minutes = (int) seconds / 60;
		this->m_seconds = modReal(seconds, (Real) 60.0);
	}else{
		this->m_hours = 0;
		this->m_minutes = 0;
		this->m_seconds = seconds;
	}
}

std::ostream& operator<<(std::ostream& stream, const TimeFrame& time){
	if(time.m_hours > 0 && time.m_minutes > 0){
		stream << time.m_hours << " h, " << time.m_minutes << " min, " << StringHelper::number2String(time.m_seconds, 2) << " sec";
	}else if(time.m_minutes > 0){
		stream <<  time.m_minutes << " min, " << StringHelper::number2String(time.m_seconds, 2) << " sec";
	}else if(time.m_seconds >= 1.0){
		stream << StringHelper::number2String(time.m_seconds, 2) << " sec";
	}else{
		stream << StringHelper::number2String(time.m_seconds * (Real) 1000.0, 3) << " milisec";
	}
	return stream;
}

TimeFrame& TimeFrame::operator*=(const Real fac){
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
