/*
 * AvgNumber.h
 *
 *  Created on: 31.01.2017
 *      Author: Max
 */

#ifndef UTILITY_AVGNUMBER_H_
#define UTILITY_AVGNUMBER_H_


class AvgNumber {
public:

	AvgNumber(): m_mean(0), m_counter(0){};

	void addNew(double val){
		++m_counter;
		const double fac = 1.0 / m_counter;
		m_mean = fac * val + (1.0-fac) * m_mean;
	}

	double mean() const { return m_mean; };

	double counter() const { return m_mean; };

private:
	double m_mean;
	double m_counter; // no convert needed
};


#endif /* UTILITY_AVGNUMBER_H_ */
