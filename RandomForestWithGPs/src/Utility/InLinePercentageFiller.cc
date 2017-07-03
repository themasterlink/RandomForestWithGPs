/*
 * InLinePercentageFiller.cc
 *
 *  Created on: 14.09.2016
 *      Author: Max
 */

#include "InLinePercentageFiller.h"
#include "curses.h"
#include "Util.h"

InLinePercentageFiller::InLinePercentageFiller(): m_max(0), m_dMax(0) {}

void InLinePercentageFiller::setActValueAndPrintLine(const long iAct){
	if(iAct <= m_max && iAct >= 0){
		setActPercentageAndPrintLine((Real) iAct / (Real) m_max * (Real) 100.0, iAct == m_max);
	}else{
//		printError("Something went wrong: " << iAct);
	}
}

void InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(const unsigned long amountOfCalcedElements, const bool lastElement){
	std::stringstream str;
	const Real seconds = m_sw.elapsedSeconds();
	const Real dAct = !lastElement ? std::min(seconds / m_dMax * (Real) 100., (Real) 100.) : (Real) 100.;
	const long amountOfElements = std::max(COLS / 2, 100);
	for(long i = 0; i < amountOfElements; ++i){
		const Real t = (Real) (i / (Real) amountOfElements * 100.);
		if(t <= dAct){
			str << "#";
		}else{
			str << " ";
		}
	}
	str << StringHelper::number2String(dAct, 2) << " %%";
	if(dAct > 0. && dAct < 100. && seconds > 0.){
		str << ", rest time is: " << TimeFrame(m_dMax - seconds);
	}else if(dAct >= 100.){
		str <<", done in: " << m_sw.elapsedAsTimeFrame();
	}
	if(amountOfCalcedElements > 0){
		str << ", " << amountOfCalcedElements << " calculated elements";
	}
	ScreenOutput::instance().printInProgressLine(str.str());
}

void InLinePercentageFiller::setActPercentageAndPrintLine(const Real dAct, const bool lastElement){
	if(dAct >= 0){
		std::stringstream str;
		const long amountOfElements = std::max(COLS / 2, 100);
		for(long i = 0; i < amountOfElements; ++i){
			const Real t = i / (Real) amountOfElements * (Real) 100.;
			if(t <= dAct){
				str << "#";
			}else{
				str << " ";
			}
		}
		str << StringHelper::number2String(dAct, 2) << " %%";
		if((dAct > 0. && dAct < 100.) || !lastElement){
			TimeFrame frame = m_sw.elapsedAsTimeFrame();
			str << ", rest time is: " << (frame * (1. / dAct * 100.) - frame);
		}else if(dAct >= 100.){
			str << ", done in: " << m_sw.elapsedAsTimeFrame();
		}
		ScreenOutput::instance().printInProgressLine(str.str());
	}
}
