//
// Created by denn_ma on 9/4/17.
//

#include "RandomExponentialNr.h"


RandomExponentialNr::RandomExponentialNr(const Real lambda, const int seed):
		m_generator(seed),
		m_exponential(lambda){
}

void RandomExponentialNr::reset(const Real lambda){
	m_exponential.param(std::exponential_distribution<Real>::param_type(lambda));
}

void RandomExponentialNr::setSeed(const int seed){
	m_generator.seed(seed);
};
