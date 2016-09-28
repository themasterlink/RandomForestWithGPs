/*
 * IVM.cc
 *
 *  Created on: 27.09.2016
 *      Author: Max
 */

#include "IVM.h"
#include <boost/math/special_functions/erf.hpp>

#define LOG2   0.69314718055994528623
#define LOG2PI 1.8378770664093453391
#define SQRT2  1.4142135623730951455

IVM::IVM(): m_dataPoints(0), m_numberOfInducingPoints(0) {
}

IVM::~IVM() {
}

void IVM::init(const Matrix& dataMat, const Vector& y, const unsigned int numberOfInducingPoints){
	m_dataMat = dataMat;
	m_y = y;
	m_dataPoints = m_dataMat.cols();
	m_numberOfInducingPoints = numberOfInducingPoints;
	m_kernel.init(m_dataMat);
}

void IVM::train(){
	if(m_kernel.calcDiagElement() == 0){
		printError("The kernel diagonal is 0, this kernel params are invalid:" + m_kernel.prettyString());
	}
	Vector m = Vector::Zero(m_dataPoints);
	Vector beta = Vector::Zero(m_dataPoints);
	Vector mu = Vector::Zero(m_dataPoints);
	Vector zeta = Vector(m_dataPoints);
	m_I.clear();
	m_J.clear();
	for(unsigned int i = 0; i < m_dataPoints; ++i){
		zeta[i] = m_kernel.calcDiagElement();
		m_J.push_back(i);
	}
	Vector g = Vector(m_numberOfInducingPoints);
	Vector nu = Vector(m_numberOfInducingPoints);
	Vector delta = Vector(m_numberOfInducingPoints);
	int amountOfOneClass = 0;
	for(unsigned int i = 0; i < m_dataPoints; ++i){
		if(m_y[i] == 1){
			amountOfOneClass += m_y[i];
		}
	}
	const double lambda = 1.0; // TODO check if the slope is 1.0
	const double bias = boost::math::cdf(boost::math::complement(m_logisticNormal, (double) amountOfOneClass / (double) m_dataPoints));
	int argmax = -1;
	for(unsigned int k = 0; k < m_numberOfInducingPoints; ++k){
		//List<Pair<int, double> > pointEntropies;
		delta[k] = -DBL_MAX;
		for(List<int>::const_iterator itOfJ = m_J.begin(); itOfJ != m_J.end(); ++itOfJ){
			const unsigned int j = *itOfJ; // actual element of J
			const double label = m_y[j];
			double g_kn, nu_kn;
			double tau = 1.0 / zeta[j];
			std::complex<double> tau_c(tau, 0);
            //double denom = std::max(abs(sqrt(tau_c * (tau_c / (lambda * lambda) + 1.))), EPSILON);
			double denom = std::max(fabs((double)(sqrt(tau_c * (tau_c / (lambda * lambda) + 1.0))).real()), EPSILON);
			double c = label * tau / denom;
			nu_kn = mu[j] / zeta[j];
			double u;
			if(nu_kn < EPSILON){
				u = c * bias;
			}else{
				u = label * nu_kn / denom + c * bias;
			}
			g_kn = c * exp(-(LOG2PI + u * u) / 2.0 - boost::math::erfc(-u / SQRT2) - LOG2);
			nu_kn = g_kn * (g_kn + u * c);
			double delta_kn = -log(1.0 - nu_kn * zeta[j]) / (2.0 * LOG2);
			// pointEntropies.append( (j, delta_ln));
			if(delta_kn > delta[k]){
				delta[k] = delta_kn;
				nu[k] = nu_kn;
				g[k] = g_kn;
				argmax = j;
			}
		}
	}
}
