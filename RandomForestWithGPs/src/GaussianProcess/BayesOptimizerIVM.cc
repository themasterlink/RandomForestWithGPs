/*
 * BayesOptimizerIVM.cc
 *
 *  Created on: 04.10.2016
 *      Author: Max
 */

#include "BayesOptimizerIVM.h"

BayesOptimizerIVM::BayesOptimizerIVM(IVM& ivm, bayesopt::Parameters param) :
	ContinuousModel(2,param),m_ivm(ivm){
	m_logZValues.clear();
}



double BayesOptimizerIVM::evaluateSample(const vectord& x){
	m_ivm.getGaussianKernel()->setHyperParams(x[0], x[1]);
	std::cout << "x: " << x[0] << ", " << x[1] << std::endl; // << " with: " << (int) x[2] << std::endl;
	StopWatch sw;
	m_ivm.train();
	std::cout << "LogZ is: " << m_ivm.m_logZ << ", needs: " << sw.elapsedAsTimeFrame() << std::endl;
	if(isnan(m_ivm.m_logZ)){
		return 5000;
	}
	m_logZValues.push_back(-m_ivm.m_logZ);
	return -m_ivm.m_logZ;
}

bool BayesOptimizerIVM::checkReachability(const vectord& query){
	return query[0] > 0.0 && query[1] > 0.0; // && query[2] > 0.0;
}


BayesOptimizerIVM::~BayesOptimizerIVM() {
	// TODO Auto-generated destructor stub
}

