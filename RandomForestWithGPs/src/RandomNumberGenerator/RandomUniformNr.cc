/*
 * RandomUniformNr.cc
 *
 *  Created on: 21.11.2016
 *      Author: Max
 */

#include "RandomUniformNr.h"

RandomUniformNr::RandomUniformNr(const int min, const int max, const int seed):
	m_generator(seed),
	m_uniform(min, max),
	m_isUsed(min != max){
}

RandomUniformNr::~RandomUniformNr() {
}


void RandomUniformNr::setSeed(const int seed){
	m_generator.seed(seed);
}

void RandomUniformNr::setMinAndMax(const int min, const int max){
	m_uniform.param(uniform_distribution_int::param_type(min, std::max(min, max)));
	m_isUsed = min != max;
}

int RandomUniformNr::operator()(){
	return m_uniform(m_generator);
}
