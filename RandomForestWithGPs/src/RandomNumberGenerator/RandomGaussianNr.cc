/*
 * RandomGaussianNr.cc
 *
 *  Created on: 09.07.2016
 *      Author: Max
 */

#include "RandomGaussianNr.h"
#include "../Utility/Util.h"
int RandomGaussianNr::counter = 0;

RandomGaussianNr::RandomGaussianNr(const Real mean, const Real sd, const int seed):
	m_generator((seed == -1 ? (counter++ * 137937): seed)),
	m_normal(mean, sd){
}

RandomGaussianNr::~RandomGaussianNr() = default;

void RandomGaussianNr::reset(const Real mean, const Real sd){
	m_normal.param(std::normal_distribution<Real>::param_type(mean,sd));
}

void RandomGaussianNr::setSeed(const int seed){
	m_generator.seed(seed);
};
