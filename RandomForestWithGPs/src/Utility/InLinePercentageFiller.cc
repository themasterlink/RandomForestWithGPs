/*
 * InLinePercentageFiller.cc
 *
 *  Created on: 14.09.2016
 *      Author: Max
 */

#include "InLinePercentageFiller.h"
#include "curses.h"
#include "Util.h"
#include "../Base/ScreenOutput.h"

long InLinePercentageFiller::m_max = 0;
Real InLinePercentageFiller::m_dMax = 0;
StopWatch InLinePercentageFiller::m_sw;

InLinePercentageFiller::InLinePercentageFiller(){
	printError("Should never be called!");
}

InLinePercentageFiller::~InLinePercentageFiller(){
	printError("Should never be called!");
}


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
	char buffer [20];
	sprintf(buffer, "| %3.2f", dAct); // all other methods are ugly
	str << buffer << " %%";
	if(dAct > 0. && dAct < 100. && seconds > 0.){
		str << ", rest time is: " << TimeFrame(m_dMax - seconds);
	}else if(dAct >= 100.){
		str <<", done in: " << m_sw.elapsedAsTimeFrame();
	}
	if(amountOfCalcedElements > 0){
		str << ", " << amountOfCalcedElements << " calculated elements";
	}
	ScreenOutput::printInProgressLine(str.str());
}

void InLinePercentageFiller::setActPercentageAndPrintLine(const Real dAct, const bool lastElement){
	if(dAct >= 0 && dAct <= 100.0){
		std::stringstream str;
		const long amountOfElements = std::max(COLS / 2, 100);
		for(long i = 0; i < amountOfElements; ++i){
			const Real t = i / (Real) amountOfElements * 100.;
			if(t <= dAct){
				str << "#";
			}else{
				str << " ";
			}
		}
		char buffer [20];
		sprintf(buffer, "| %3.2f", dAct); // all other methods are ugly
		str << buffer << " %%";
		if((dAct > 0. && dAct < 100.) || !lastElement){
			TimeFrame frame = m_sw.elapsedAsTimeFrame();
			str << ", rest time is: " << (frame * (1. / dAct * 100.) - frame);
		}else if(dAct >= 100.|| lastElement){
			str << ", done in: " << m_sw.elapsedAsTimeFrame();
		}
		ScreenOutput::printInProgressLine(str.str());
	}else{
//		printError("The value is not in percent: " << dAct);
	}
}
