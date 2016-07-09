/*
 * RandomGaussianNr.cc
 *
 *  Created on: 09.07.2016
 *      Author: Max
 */

#include "RandomGaussianNr.h"

RandomGaussianNr::RandomGaussianNr(const double mean, const double sd):
	m_normalGenerator(m_generator, normal_distribution(mean, sd)){
}

RandomGaussianNr::~RandomGaussianNr(){
}


void RandomGaussianNr::reset(const double mean, const double sd){
	m_normalGenerator.distribution().param(normal_distribution::param_type(mean,sd));
}
