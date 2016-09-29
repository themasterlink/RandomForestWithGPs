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

IVM::IVM(): m_dataPoints(0), m_numberOfInducingPoints(0), m_bias(0), m_lambda(0) {
}

IVM::~IVM() {
}

void IVM::init(const Matrix& dataMat, const Vector& y, const unsigned int numberOfInducingPoints){
	m_dataMat = dataMat;
	m_y = y;
	m_dataPoints = m_dataMat.cols();
	m_numberOfInducingPoints = numberOfInducingPoints;
	m_kernel.init(m_dataMat);
	int amountOfOneClass = 0;
	for(unsigned int i = 0; i < m_dataPoints; ++i){
		if(m_y[i] == 1){
			amountOfOneClass += m_y[i];
		}
	}
	m_bias = boost::math::cdf(boost::math::complement(m_logisticNormal, (double) amountOfOneClass / (double) m_dataPoints));
	m_lambda = 1.0; // TODO check if the slope is 1.0
	m_nuTilde = Vector(m_numberOfInducingPoints);
	m_tauTilde = Vector(m_numberOfInducingPoints);
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
	for(unsigned int k = 0; k < m_numberOfInducingPoints; ++k){
		int argmax = -1;
		//List<Pair<int, double> > pointEntropies;
		delta[k] = -DBL_MAX;
		for(List<int>::const_iterator itOfJ = m_J.begin(); itOfJ != m_J.end(); ++itOfJ){
			const unsigned int j = *itOfJ; // actual element of J
			const double label = m_y[j];
			double g_kn, nu_kn;
			double tau = 1.0 / zeta[j];
			std::complex<double> tau_c(tau, 0);
            //double denom = std::max(abs(sqrt(tau_c * (tau_c / (lambda * lambda) + 1.))), EPSILON);
			double denom = std::max(fabs((double)(sqrt(tau_c * (tau_c / (m_lambda * m_lambda) + 1.0))).real()), EPSILON);
			double c = label * tau / denom;
			nu_kn = mu[j] / zeta[j];
			double u;
			if(nu_kn < EPSILON){
				u = c * m_bias;
			}else{
				u = label * nu_kn / denom + c * m_bias;
			}
			g_kn = c * exp(-(LOG2PI + u * u) / 2.0 - boost::math::erfc(-u / SQRT2) - LOG2);
			nu_kn = g_kn * (g_kn + u * c);
			double delta_kn = -log(1.0 - nu_kn * (double)zeta[j]) / (2.0 * LOG2);
			// pointEntropies.append( (j, delta_ln));
			if(delta_kn > delta[k]){
				delta[k] = delta_kn;
				nu[k] = nu_kn;
				g[k] = g_kn;
				argmax = j;
			}
		}
		if(argmax == -1){
			printError("No new inducing point was found!");
			break;
		}
		// refine site params, posterior params & M, L, K
		if(fabs((double)nu[k]) > EPSILON){
			m[argmax] = g[k] / nu[k] + mu[argmax];
		}
		beta[argmax] = nu[k] / (1.0 - nu[k] * zeta[argmax]);
		if(beta[argmax] < EPSILON){
			beta[argmax] = EPSILON;
		}
		Vector s_nk, a_nk, k_nk = Vector(m_dataPoints);
		for(unsigned int i = 0; i < m_dataPoints; ++i){
			k_nk[i] = m_kernel.kernelFunc(i, argmax);
		}
		if(k == 0){
			s_nk = k_nk;
		}else{
			Vector colVec = m_M.col(argmax);
			s_nk = k_nk - (colVec.transpose() * m_M).transpose();
		}
		zeta -= nu[k] * s_nk.cwiseProduct(s_nk);
		mu += g[k] * s_nk;
		const double sqrtNu = sqrt((double)nu[k]);
		if(argmax < m_M.cols()){
			a_nk = m_M.col(argmax);
		}
		if(k == 0){
			m_K = Matrix(1,1);
			m_K(0,0) = m_kernel.calcDiagElement();
			m_L = Matrix(1,1);
			m_L(0,0) = 1.0 / sqrtNu;
		}else{
			Vector k_vec = Vector(m_I.size());
			unsigned int t = 0;
			for(List<int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI){
				k_vec[t++] = m_kernel.kernelFunc(*itOfI, argmax);
			}
			Matrix D(m_K.rows() + 1, m_K.cols() + 1);
			D << m_K, k_vec,
			     k_vec.transpose(), m_kernel.calcDiagElement();
			m_K = D;
			// update L
			Matrix D2(m_L.rows() + 1, m_L.cols() + 1);
			D2 << m_L, Vector::Zero(k),
					a_nk.transpose(), 1. / sqrtNu;
			m_L = D2;
		}

		// update M
		if(k == 0){
			m_M = Matrix(1, m_dataPoints);
			for(unsigned int i = 0; i < m_dataPoints; ++i){
				m_M(0,i) = sqrtNu * s_nk[i];
			}
		}else{
			Matrix D(m_M.rows() + 1, m_M.cols());
			D << m_M,
				(sqrtNu * s_nk).transpose();
			m_M = D;
		}

		m_I.push_back(argmax);
		m_J.remove(argmax);
	}
	unsigned int l = 0;
	for(List<int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI){
		m_nuTilde[l] = m[*itOfI] * beta[*itOfI];
		m_tauTilde[l] = beta[*itOfI];
	}
	m_choleskyLLT.compute(m_L);

}

double IVM::predict(const Vector& input){
	const unsigned int n = m_I.size();
	Vector k_star(n);
	unsigned int i = 0;
	for(List<int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI, ++i){
		k_star[i] = m_kernel.kernelFuncVec(input, m_dataMat.col(*itOfI));
	}
	Vector v = m_choleskyLLT.solve(k_star);
	Vector mu_tilde = m_nuTilde.cwiseQuotient(m_tauTilde);
	double mu_star = (mu_tilde + m_bias * Vector::Ones(n)).dot(v);
	double sigma_star = (m_kernel.calcDiagElement() - k_star.dot(v));
	return boost::math::erfc(-(mu_star / sqrt(1.0 /(m_lambda * m_lambda) + sigma_star)) / SQRT2) / 2.0;
}
