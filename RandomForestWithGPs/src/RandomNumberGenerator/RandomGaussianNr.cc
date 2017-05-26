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
	m_normalGenerator(nullptr),
	m_mean(mean),
	m_sd(sd){
	m_normalGenerator = std::make_unique<variante_generator>(m_generator, normal_distribution(m_mean, m_sd));
}

RandomGaussianNr::~RandomGaussianNr() = default;

void RandomGaussianNr::reset(const Real mean, const Real sd){
	m_normalGenerator->distribution().param(normal_distribution::param_type(mean,sd));
	m_mean = mean;
	m_sd = sd;
}

void RandomGaussianNr::setSeed(const int seed){
	m_generator.seed(seed);
	m_normalGenerator = std::make_unique<variante_generator>(m_generator, normal_distribution(m_mean, m_sd));
};
