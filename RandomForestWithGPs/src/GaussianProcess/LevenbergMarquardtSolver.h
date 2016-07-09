/*
 * LevenbergMarquardSolver.h
 *
 *  Created on: 15.06.2016
 *      Author: Max
 */

#ifndef LEVENBERGMARQUARDTSOLVER_H_
#define LEVENBERGMARQUARDTSOLVER_H_

#include "../Utility/Util.h"
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include "GaussianProcessBinaryClass.h"

struct OptimizeFunctor
{
  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const{
      if(m_gp != NULL){
  		std::cout << "x: " << x.transpose() << std::endl;
		m_gp->trainLM(m_logZ, m_dLogZ);
		std::cout << "m_logZ: " << m_logZ << std::endl;
		std::cout << "m_dLogZ: " << m_dLogZ[0] << ", "<< m_dLogZ[1] << ", "<< m_dLogZ[2]<< std::endl;
		fvec(0) = m_logZ;
		fvec(1) = m_logZ;
		fvec(2) = m_logZ;
	}
    return 0;
  }

  int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const{
	  std::cout << "x: " << x.transpose() << std::endl;
	  fjac = Eigen::MatrixXd::Zero(3,3);
	  for(int i = 0; i < 3; ++i){
		  fjac(i,i) = m_dLogZ[i];
		  std::cout << "df is called!" << std::endl;
	  }
	  return 0;
  }

  OptimizeFunctor(GaussianProcessBinaryClass* gp): m_logZ(0.0){
	  m_dLogZ.reserve(3);
	  m_gp = gp;
  }

  GaussianProcessBinaryClass* m_gp;
  mutable std::vector<double> m_dLogZ;
  mutable double m_logZ;
  int inputs() const { return 3; }
  int values() const { return 3; } // number of constraints
};

#endif /* LEVENBERGMARQUARDTSOLVER_H_ */
