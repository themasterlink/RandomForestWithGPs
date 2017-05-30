/*
 * AvgNumber.h
 *
 *  Created on: 31.01.2017
 *      Author: Max
 */

#ifndef UTILITY_AVGNUMBER_H_
#define UTILITY_AVGNUMBER_H_

#include "../Base/BaseType.h"

class AvgNumber {
public:

	AvgNumber(): m_mean(0), m_counter(0){};

	void addNew(Real val){
		++m_counter;
		const Real fac = (Real) 1.0 / m_counter;
		m_mean = fac * val + ((Real) 1.0 - fac) * m_mean;
	}

	Real mean() const { return m_mean; };

	Real counter() const { return m_mean; };

private:
	Real m_mean;
	Real m_counter; // no convert needed
};


#endif /* UTILITY_AVGNUMBER_H_ */
