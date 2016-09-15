/*
 * InLinePercentageFiller.h
 *
 *  Created on: 14.09.2016
 *      Author: Max
 */

#ifndef UTILITY_INLINEPERCENTAGEFILLER_H_
#define UTILITY_INLINEPERCENTAGEFILLER_H_

class InLinePercentageFiller {
public:

	static void setActMax(const int iMax);

	static void setActValueAndPrintLine(const int iAct);

	static void setActPercentageAndPrintLine(const double iAct);

private:
	InLinePercentageFiller();
	virtual ~InLinePercentageFiller();

	static int m_max;
};

inline
void InLinePercentageFiller::setActMax(const int iMax){
	m_max = iMax;
}

#endif /* UTILITY_INLINEPERCENTAGEFILLER_H_ */
