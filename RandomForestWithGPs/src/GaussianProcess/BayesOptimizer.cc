/*
 * BayesOptimizer.cc
 *
 *  Created on: 09.07.2016
 *      Author: Max
 */

#include "BayesOptimizer.h"

BayesOptimizer::BayesOptimizer(GaussianProcess& gp, bayesopt::Parameters param):
	ContinuousModel(2,param), m_gp(gp), m_lowestValue(-100){ //, bestVal(-100000000), bestLen(0), bestSigma(0) {
}

BayesOptimizer::~BayesOptimizer() {
	// TODO Auto-generated destructor stub
}

double BayesOptimizer::evaluateSample(const vectord& x) {
	GaussianProcess::Status status(GaussianProcess::NANORINFERROR);
	double logZ = 0.0;
	if(x[0] * x[1] < 50){ // avoid overfitting!
		m_gp.getKernel().setHyperParams(x[0], x[1], m_gp.getKernel().sigmaN());
		status = m_gp.trainBayOpt(logZ,1);
		if(logZ > 1.0){ // avoid overfitting!
			logZ = m_lowestValue;
		}
		if(logZ < m_lowestValue){
			m_lowestValue = logZ;
		}
	}
	/*std::cout << RED << "logZ: " << logZ << RESET << std::endl;
	if(logZ > bestVal){
		bestVal = logZ;
		bestLen = m_gp.getKernel().len();
		bestSigma = m_gp.getKernel().sigmaF();
		//getchar();
	}*/
	return status == GaussianProcess::NANORINFERROR ? 10000 - m_lowestValue : 10000 - logZ;
};


bool BayesOptimizer::checkReachability(const vectord& query) {
	if(query[0] > 0.0 && query[0] < 5 * fabs(m_gp.getLenMean()) && query[1] > 0.0){
		return true;
	}
	return false;
};
