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

	SINGELTON_MACRO(InLinePercentageFiller);

public:

	void setActMax(const long iMax);

	void setActMaxTime(const Real dMax);

	void setActValueAndPrintLine(const long iAct);

	void printLineWithRestTimeBasedOnMaxTime(const unsigned long amountOfCalcedElements, const bool lastElement = false);

	void setActPercentageAndPrintLine(const Real dAct, const bool lastElement = false);

private:

	long m_max;

	Real m_dMax;

	StopWatch m_sw;
};

inline
void InLinePercentageFiller::setActMax(const long iMax){
	m_max = iMax - 1;
	m_sw.startTime();
}

inline
void InLinePercentageFiller::setActMaxTime(const Real dMax){
	m_dMax = dMax;
	m_sw.startTime();
}

#endif /* UTILITY_INLINEPERCENTAGEFILLER_H_ */
