/*
 * TimeFrame.h
 *
 *  Created on: 07.06.2016
 *      Author: Max
 */

#ifndef UTILITY_TIMEFRAME_H_
#define UTILITY_TIMEFRAME_H_

#include <cmath>
#include <iostream>

class TimeFrame {
public:
	TimeFrame(const double seconds);

	void setWithSeconds(const double seconds);

	inline double getSeconds() const;

	TimeFrame& operator*=(const double fac);

	TimeFrame& operator+=(const TimeFrame& rhs);

	TimeFrame& operator-=(const TimeFrame& rhs);

	inline double getOnlySeconds() const{ return m_seconds; }

	inline int getOnlyMinutes() const{ return m_minutes; }

	inline int getOnlyHours() const{ return m_hours; }

	friend std::ostream& operator<<(std::ostream& stream, const TimeFrame& time);

	friend const TimeFrame operator*(const TimeFrame& frame, const double fac);

	friend const TimeFrame operator-(const TimeFrame& lhs, const TimeFrame& rhs);

	friend const TimeFrame operator+(const TimeFrame& lhs, const TimeFrame& rhs);
private:
	double m_seconds;
	int m_minutes;
	int m_hours;
};

inline double TimeFrame::getSeconds() const{
	return m_seconds + (m_minutes * 60 + m_hours * 3600);
}

#endif /* UTILITY_TIMEFRAME_H_ */