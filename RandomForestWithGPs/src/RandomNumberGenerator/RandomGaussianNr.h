/*
 * RandomGaussianNr.h
 *
 *  Created on: 09.07.2016
 *      Author: Max
 */

#ifndef RANDOMNUMBERGENERATOR_RANDOMGAUSSIANNR_H_
#define RANDOMNUMBERGENERATOR_RANDOMGAUSSIANNR_H_

#include "../Utility/Util.h"

#ifdef BUILD_OLD_CODE
class GaussianProcessWriter;
#endif // BUILD_OLD_CODE

class RandomGaussianNr{
#ifdef BUILD_OLD_CODE
	friend GaussianProcessWriter;
#endif // BUILD_OLD_CODE
public:

	RandomGaussianNr(const Real mean = (Real) 0.0, const Real sd = (Real) 1.0, const int seed = -1);
	virtual ~RandomGaussianNr();

	void reset(const Real mean, const Real sd);

	// get next Number
	Real operator()();

	void setSeed(const int seed);

private:
	static int counter;

	GeneratorType m_generator;
	std::normal_distribution<Real> m_normal;
};

inline
Real RandomGaussianNr::operator()(){
	return (Real) m_normal(m_generator);
}

#endif /* RANDOMNUMBERGENERATOR_RANDOMGAUSSIANNR_H_ */
