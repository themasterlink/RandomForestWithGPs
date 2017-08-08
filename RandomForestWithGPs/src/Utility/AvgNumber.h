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

	AvgNumber() = default;

	explicit AvgNumber(Real startVal): m_mean(startVal){};

	void addNew(Real val){
		++m_counter;
		const Real fac = (Real) 1.0 / (Real) m_counter;
		m_mean = fac * val + ((Real) 1.0 - fac) * m_mean;
	}

	Real mean() const { return m_mean; };

	unsigned long counter() const { return m_counter; };

	void reset() { m_counter = 0; }

private:
	Real m_mean{0};
	unsigned long m_counter{0}; // no convert needed
};


#endif /* UTILITY_AVGNUMBER_H_ */
