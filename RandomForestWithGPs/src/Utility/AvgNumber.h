/*
 * AvgNumber.h
 *
 *  Created on: 31.01.2017
 *      Author: Max
 */

#ifndef UTILITY_AVGNUMBER_H_
#define UTILITY_AVGNUMBER_H_

#include "../Base/BaseType.h"

/**
 * \brief Calculates the average number of the given inputs, only saves the mean and the counter
 *  \f[ mean = \frac{1}{counter} \cdot newVal + \left( 1 - \frac{1}{counter} \right) \cdot mean \f]
 */
class AvgNumber {
public:
	/**
	 * \brief Init the average number with mean  0 and the counter with 0
	 * 			The first element set the average to the given value, after that the average is always calculated
	 * 			between the existing mean and the new value
	 */
	AvgNumber() = default;

	/**
	 * \brief Init the average number with the given value instead of with 0, does not count as the first value!
	 * \param startVal Only the start value, not the first value of the series!
	 */
	explicit AvgNumber(Real startVal): m_mean(startVal){};

	/**
	 * \brief Add a new value, updates the mean and the counter
	 * \param val the new value
	 */
	void addNew(Real val){
		++m_counter;
		const Real fac = (Real) 1.0 / (Real) m_counter;
		m_mean = fac * val + ((Real) 1.0 - fac) * m_mean;
	}

	/**
	 * \brief Return the current mean (average)
	 * \return the current mean
	 */
	Real mean() const { return m_mean; };

	/**
	 * \brief Return the counter, how many numbers have been added
	 * \return the counter
	 */
	unsigned long counter() const { return m_counter; };

	/**
	 * \brief Resets the average number and it starts from the beginning
	 */
	void reset() { m_counter = 0; }

private:
	/**
	 * \brief The mean, which saves the average number
	 */
	Real m_mean{0};
	/**
	 * \brief Counter: how many numbers where added
	 */
	unsigned long m_counter{0};
};


#endif /* UTILITY_AVGNUMBER_H_ */
