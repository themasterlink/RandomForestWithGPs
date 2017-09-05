//
// Created by denn_ma on 9/4/17.
//

#ifndef RANDOMFORESTWITHGPS_RANDOMEXPONENTIALNR_H
#define RANDOMFORESTWITHGPS_RANDOMEXPONENTIALNR_H

#include "../Utility/Util.h"

class RandomExponentialNr {

public:
	RandomExponentialNr(const Real lambda, const int seed);

	virtual ~RandomExponentialNr(){};

	void reset(const Real lambda);

	// get next Number
	Real operator()();

	void setSeed(const int seed);

private:

	GeneratorType m_generator;
	std::exponential_distribution<Real> m_exponential;
};


inline
Real RandomExponentialNr::operator()(){
	return m_exponential(m_generator);
}


#endif //RANDOMFORESTWITHGPS_RANDOMEXPONENTIALNR_H
