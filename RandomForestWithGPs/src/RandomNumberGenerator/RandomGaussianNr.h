/*
 * RandomGaussianNr.h
 *
 *  Created on: 09.07.2016
 *      Author: Max
 */

#ifndef RANDOMNUMBERGENERATOR_RANDOMGAUSSIANNR_H_
#define RANDOMNUMBERGENERATOR_RANDOMGAUSSIANNR_H_

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

class RandomGaussianNr{
public:

	typedef boost::random::mt19937 base_generator_type; // generator type
	typedef boost::random::normal_distribution<> normal_distribution;
	typedef boost::random::variate_generator<base_generator_type&, normal_distribution > variante_generator;

	RandomGaussianNr(const double mean = 0.0, const double sd = 1.0);
	virtual ~RandomGaussianNr();

	void reset(const double mean, const double sd);

	// get next Number
	double operator()();

private:
	base_generator_type m_generator;
	variante_generator m_normalGenerator;
};

inline
double RandomGaussianNr::operator()(){
	return m_normalGenerator();
}

#endif /* RANDOMNUMBERGENERATOR_RANDOMGAUSSIANNR_H_ */
