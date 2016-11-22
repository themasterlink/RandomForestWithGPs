/*
 * RandomUniformNr.h
 *
 *  Created on: 21.11.2016
 *      Author: Max
 */

#ifndef RANDOMNUMBERGENERATOR_RANDOMUNIFORMNR_H_
#define RANDOMNUMBERGENERATOR_RANDOMUNIFORMNR_H_

#include <boost/random.hpp>
#include <boost/random/uniform_int.hpp>

class RandomUniformNr {
public:

	typedef boost::random::mt19937 base_generator_type; // generator type
	typedef boost::random::uniform_int_distribution<int> uniform_distribution_int; // generator type

	RandomUniformNr(const int min, const int max, const int seed);
	virtual ~RandomUniformNr();

	void setSeed(const int seed);

	void setMinAndMax(const int min, const int max);

	// returns true if min and max are not equal
	bool isUsed() const{ return m_isUsed; };

	int operator()();

private:
	base_generator_type m_generator;

	uniform_distribution_int m_uniform;

	bool m_isUsed;

};

#endif /* RANDOMNUMBERGENERATOR_RANDOMUNIFORMNR_H_ */
