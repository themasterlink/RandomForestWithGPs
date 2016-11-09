/*
 * RandomGaussianNr.cc
 *
 *  Created on: 09.07.2016
 *      Author: Max
 */

#include "RandomGaussianNr.h"

int RandomGaussianNr::counter = 0;

RandomGaussianNr::RandomGaussianNr(const double mean, const double sd, const int seed):
	m_generator((seed == -1 ? (counter++ * 137937): seed)),
	m_normalGenerator(m_generator, normal_distribution(mean, sd)),
	m_mean(mean),
	m_sd(sd){
}

RandomGaussianNr::~RandomGaussianNr(){
}


void RandomGaussianNr::reset(const double mean, const double sd){
	m_normalGenerator.distribution().param(normal_distribution::param_type(mean,sd));
	m_mean = mean;
	m_sd = sd;
}
