/*
 * RandomGaussianNr.h
 *
 *  Created on: 09.07.2016
 *      Author: Max
 */

#ifndef RANDOMNUMBERGENERATOR_RANDOMGAUSSIANNR_H_
#define RANDOMNUMBERGENERATOR_RANDOMGAUSSIANNR_H_

#include "../Utility/Util.h"
#include <boost/random/normal_distribution.hpp>

class GaussianProcessWriter;

class RandomGaussianNr{
	friend GaussianProcessWriter;
public:

	using base_generator_type = GeneratorType; // generator type
	using normal_distribution = boost::random::normal_distribution<>;
	using variante_generator = boost::random::variate_generator<base_generator_type&, normal_distribution >;

	RandomGaussianNr(const Real mean = (Real) 0.0, const Real sd = (Real) 1.0, const int seed = -1);
	virtual ~RandomGaussianNr();

	void reset(const Real mean, const Real sd);

	// get next Number
	Real operator()();

	void setSeed(const int seed);

private:
	static int counter;

	base_generator_type m_generator;
	std::unique_ptr<variante_generator> m_normalGenerator;
	Real m_mean;
	Real m_sd;
};

inline
Real RandomGaussianNr::operator()(){
	return (*m_normalGenerator)();
}

#endif /* RANDOMNUMBERGENERATOR_RANDOMGAUSSIANNR_H_ */
