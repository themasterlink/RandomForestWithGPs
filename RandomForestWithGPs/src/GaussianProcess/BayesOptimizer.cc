/*
 * BayesOptimizer.cc
 *
 *  Created on: 09.07.2016
 *      Author: Max
 */

#include "BayesOptimizer.h"

BayesOptimizer::BayesOptimizer(GaussianProcessBinary& gp, bayesopt::Parameters param):
	ContinuousModel(2,param), m_gp(gp) {
}

BayesOptimizer::~BayesOptimizer() {
	// TODO Auto-generated destructor stub
}

double BayesOptimizer::evaluateSample(const vectord& x) {
	printWarning("X: " << x);
	m_gp.getKernel().setHyperParams(x[0], x[1], m_gp.getKernel().sigmaN());
	std::vector<double> dLogZ;
	dLogZ.reserve(3);
	double logZ = 0.0;
	GaussianProcessBinary::Status status = m_gp.trainLM(logZ, dLogZ);
	return status == GaussianProcessBinary::NANORINFERROR ? -std::numeric_limits<double>::infinity() : logZ;
};


bool BayesOptimizer::checkReachability(const vectord& query) {
	if(query[0] > 0.0 && query[0] < 5 * fabs(m_gp.getLenMean()) && query[1] > 0.0){
		return true;
	}
	return false;
};
