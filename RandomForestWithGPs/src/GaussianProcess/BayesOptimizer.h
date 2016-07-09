/*
 * BayesOptimizer.h
 *
 *  Created on: 09.07.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_BAYESOPTIMIZER_H_
#define GAUSSIANPROCESS_BAYESOPTIMIZER_H_

#include <bayesopt/bayesopt.hpp>
#include <bayesopt/bayesoptbase.hpp>
#include <bayesopt/parameters.hpp>
#include "../Data/Data.h"
#include "GaussianProcessBinary.h"
#include <limits>

class BayesOptimizer : public bayesopt::ContinuousModel {

public:
	BayesOptimizer(GaussianProcessBinary& gp, bayesopt::Parameters param);

	double evaluateSample(const vectord& x);

	bool checkReachability(const vectord& query);

	virtual ~BayesOptimizer();

private:
	GaussianProcessBinary& m_gp;
};

#endif /* GAUSSIANPROCESS_BAYESOPTIMIZER_H_ */
