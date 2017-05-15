/*
 * BayesOptimizer.cc
 *
 *  Created on: 09.07.2016
 *      Author: Max
 */

#include "BayesOptimizer.h"

BayesOptimizer::BayesOptimizer(GaussianProcess& gp, bayesopt::Parameters param):
	ContinuousModel(2,param), m_gp(gp), m_lowestValue(-100000), m_worstValue(100000){ //, bestVal(-100000000), bestLen(0), bestSigma(0) {
}

BayesOptimizer::~BayesOptimizer() {
	// TODO Auto-generated destructor stub
}

double BayesOptimizer::evaluateSample(const vectord& x) {
	GaussianProcess::Status status(GaussianProcess::Status::NANORINFERROR);
	double logZ = 0.0;
	const double upperBound = 10000;
	if(x[0] * x[1] < 50){ // avoid overfitting!
		m_gp.getKernel().setHyperParams(x[0], x[1]);
		status = m_gp.trainBayOpt(logZ,1);
		if(logZ > upperBound){
			printError("The upper bound is to low, the result of the Bayessian Optimiziation can be wrong!");
		}
		if(logZ > 1.0){ // avoid overfitting!
			logZ = m_lowestValue;
		}
		if(logZ > m_lowestValue){
			m_lowestValue = logZ;
		}
		if(logZ < m_worstValue){
			m_worstValue = logZ;
		}
	}
	/*std::cout << RED << "logZ: " << logZ << RESET << std::endl;
	if(logZ > bestVal){
		bestVal = logZ;
		bestLen = m_gp.getKernel().len();
		bestSigma = m_gp.getKernel().sigmaF();
		//getchar();
	}*/
	return status == GaussianProcess::Status::NANORINFERROR ? upperBound - m_worstValue : upperBound - logZ;
};


bool BayesOptimizer::checkReachability(const vectord& query) {
	if(query[0] > 0.0 && query[1] > 0.0){
		return true;
	}
	return false;
};
