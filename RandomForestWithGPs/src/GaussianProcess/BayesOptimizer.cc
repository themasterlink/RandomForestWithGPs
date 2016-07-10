/*
 * BayesOptimizer.cc
 *
 *  Created on: 09.07.2016
 *      Author: Max
 */

#include "BayesOptimizer.h"

BayesOptimizer::BayesOptimizer(GaussianProcessBinary& gp, bayesopt::Parameters param):
	ContinuousModel(2,param), m_gp(gp), m_lowestValue(1000000), bestVal(-100000000), bestLen(0), bestSigma(0) {
}

BayesOptimizer::~BayesOptimizer() {
	// TODO Auto-generated destructor stub
}

double BayesOptimizer::evaluateSample(const vectord& x) {
	printWarning("X: " << x);
	m_gp.getKernel().setHyperParams(x[0], x[1], m_gp.getKernel().sigmaN());
	double logZ = 0.0;
	GaussianProcessBinary::Status status = m_gp.trainBayOpt(logZ,1);
	std::cout << RED << "logZ: " << logZ << RESET << std::endl;
	if(-logZ < m_lowestValue){
		m_lowestValue = -logZ;
	}
	if(logZ > bestVal){
		bestVal = logZ;
		bestLen = m_gp.getKernel().len();
		bestSigma = m_gp.getKernel().sigmaF();
		//getchar();
	}
	std::cout << RED << "bestVal: " << bestVal << ", kernel: " << bestLen << ", " << bestSigma << RESET << std::endl;

	return status == GaussianProcessBinary::NANORINFERROR ? m_lowestValue : 10000 - logZ;
};


bool BayesOptimizer::checkReachability(const vectord& query) {
	if(query[0] > 0.0 && query[0] < 5 * fabs(m_gp.getLenMean()) && query[1] > 0.0){
		return true;
	}
	return false;
};