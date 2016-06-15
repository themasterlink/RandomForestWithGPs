/*
 * GaussianProcessBinaryClass.cc
 *
 *  Created on: 15.06.2016
 *      Author: Max
 */

#include "GaussianProcessBinaryClass.h"

GaussianProcessBinaryClass::GaussianProcessBinaryClass()
{
	// TODO Auto-generated constructor stub

}

GaussianProcessBinaryClass::~GaussianProcessBinaryClass()
{
	// TODO Auto-generated destructor stub
}

void GaussianProcessBinaryClass::train(const int dataPoints, const Eigen::MatrixXd& K, const Eigen::VectorXd& y){
	m_f = Eigen::VectorXd::Zero(dataPoints); 						// f <-- init with zeros
	const Eigen::MatrixXd eye(Eigen::MatrixXd::Identity(dataPoints,dataPoints));
	bool converged = false;
	int i = 0;
	Eigen::MatrixXd L;
	Eigen::VectorXd a;
	m_pi = Eigen::VectorXd::Zero(dataPoints);
	m_dLogPi = Eigen::VectorXd::Zero(dataPoints);
	m_ddLogPi = Eigen::VectorXd::Zero(dataPoints);
	m_sqrtDDLogPi = Eigen::VectorXd::Zero(dataPoints);
	while(!converged){
		const Eigen::VectorXd lastF = m_f;													// lastF <- save f for converge controll
		// calc - log p(y_i| f_i) -> -
		for(int i = 0; i < dataPoints; ++i){
			m_pi[i] = 1.0 / (1.0 + exp((double) -y[i] * (double) m_f[i]));
			m_dLogPi[i] = (((double)y[i] + 1.0) / 2.0) - m_pi[i];
			m_ddLogPi[i] = -(-m_pi[i] * (1 - m_pi[i])); // first minus to get -ddlog(p_i|f_i)
			m_sqrtDDLogPi[i] = sqrt((double) m_ddLogPi[i]);
		}
		//std::cout << "dP_y_i_on_fi: \n" << dP_y_i_on_fi.transpose() << std::endl;
		//std::cout << "ddP_y_i_on_fi: \n" << ddP_y_i_on_fi.transpose() << std::endl;
		//std::cout << "sqrtDDP_y_i_on_fi: \n" << sqrtDDP_y_i_on_fi.transpose() << std::endl;

		DiagMatrixXd W(m_ddLogPi);
		DiagMatrixXd WSqrt(m_sqrtDDLogPi);
		Eigen::MatrixXd C = eye + WSqrt * K * WSqrt;
		// get cholesky from C
		L = Eigen::LLT<Eigen::MatrixXd>(C).matrixL();
		const Eigen::VectorXd b = W * m_f + m_dLogPi;
		const Eigen::MatrixXd nenner = L.triangularView<Eigen::Lower>().solve(WSqrt * K * b);
		a = b - WSqrt * L.transpose().triangularView<Eigen::Upper>().solve(nenner);
		//std::cout << "a: \n" << a.transpose() << std::endl;
		m_f = K * a;
		std::cout << "f: \n" << m_f.transpose() << std::endl;
		std::cout << "diff after " << i << ": " << fabs((m_f-lastF).mean())  << std::endl;
		converged = fabs((m_f-lastF).mean()) < 0.0001;
		++i;
	}

	// marginal likelihood?
	/*Eigen::VectorXd minusLogPi;
	for(int i = 0; i < dataPoints; ++i){
		minusLogPi[i] = -log(1.0 / (1.0 + exp((double) -y[i] * (double) f[i]))); // check if this is the probability!
	}
	Eigen::VectorXd logQ = -0.5 * (a.transpose() * f) + minusLogPi -
	*/
}
