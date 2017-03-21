/*
 * InLinePercentageFiller.h
 *
 *  Created on: 14.09.2016
 *      Author: Max
 */

#ifndef UTILITY_INLINEPERCENTAGEFILLER_H_
#define UTILITY_INLINEPERCENTAGEFILLER_H_

#include "StopWatch.h"

class InLinePercentageFiller {
public:

	static void setActMax(const long iMax);

	static void setActMaxTime(const double dMax);

	static void setActValueAndPrintLine(const long iAct);

	static void printLineWithRestTimeBasedOnMaxTime(const unsigned long amountOfCalcedElements, const bool lastElement = false);

	static void setActPercentageAndPrintLine(const double dAct, const bool lastElement = false);

private:
	InLinePercentageFiller();
	virtual ~InLinePercentageFiller();

	static long m_max;

	static double m_dMax;

	static StopWatch m_sw;
};

inline
void InLinePercentageFiller::setActMax(const long iMax){
	m_max = iMax - 1;
	m_sw.startTime();
}

inline
void InLinePercentageFiller::setActMaxTime(const double dMax){
	m_dMax = dMax;
	m_sw.startTime();
}

#endif /* UTILITY_INLINEPERCENTAGEFILLER_H_ */
