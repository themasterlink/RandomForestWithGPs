/*
 * InLinePercentageFiller.cc
 *
 *  Created on: 14.09.2016
 *      Author: Max
 */

#include "InLinePercentageFiller.h"
#include "Util.h"

int InLinePercentageFiller::m_max = 0;


InLinePercentageFiller::InLinePercentageFiller(){
	printError("Should never be called!");
}

InLinePercentageFiller::~InLinePercentageFiller(){
	printError("Should never be called!");
}


void InLinePercentageFiller::setActValueAndPrintLine(const int iAct){
	if(iAct <= m_max && iAct >= 0){
		setActPercentageAndPrintLine((double) iAct / (double) m_max * 100.0, iAct == m_max);
	}else{
		printError("Something went wrong!");
	}
}

void InLinePercentageFiller::setActPercentageAndPrintLine(const double dAct, const bool lastElement){
	if(dAct >= 0 && dAct <= 100.0){
		std::cout << "\r                                                                                    ";
		std::cout << "\r";
		for(int i = 0; i < 100; ++i){
			if(i <= dAct){
				std::cout << "#";
			}else{
				std::cout << " ";
			}
		}
		printf("|  %3.2f %%", dAct);
		flush(std::cout);
		if(lastElement){
			std::cout << std::endl;
		}
	}else{
		printError("The value is not in percent: " << dAct);
	}
}
