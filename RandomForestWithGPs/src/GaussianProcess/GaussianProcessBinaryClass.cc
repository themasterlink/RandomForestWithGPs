/*
 * GaussianProcessBinaryClass.cc
 *
 *  Created on: 15.06.2016
 *      Author: Max
 */

#include "GaussianProcessBinaryClass.h"
#include "GaussianProcessMultiClass.h"

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
	int j = 0;
	Eigen::VectorXd a;
	m_pi = Eigen::VectorXd::Zero(dataPoints);
	m_dLogPi = Eigen::VectorXd::Zero(dataPoints);
	m_ddLogPi = Eigen::VectorXd::Zero(dataPoints);
	m_sqrtDDLogPi = Eigen::VectorXd::Zero(dataPoints);
	Eigen::VectorXd lastF = m_f;													// lastF <- save f for converge controll
	const Eigen::VectorXd t = (y + Eigen::VectorXd::Ones(y.rows())) * 0.5;
	double lastObjective = 100000000000;
	while(!converged){
		// calc - log p(y_i| f_i) -> -
		for(int i = 0; i < dataPoints; ++i){
			m_pi[i] = 1.0 / (1.0 + exp((double) -y[i] * (double) m_f[i]));
			m_dLogPi[i] = t[i] - m_pi[i];
			m_ddLogPi[i] = -(-m_pi[i] * (1 - m_pi[i])); // first minus to get -ddlog(p_i|f_i)
			m_sqrtDDLogPi[i] = sqrt((double) m_ddLogPi[i]);
		}

		//std::cout << "dP_y_i_on_fi: \n" << dP_y_i_on_fi.transpose() << std::endl;
		//std::cout << "ddP_y_i_on_fi: \n" << ddP_y_i_on_fi.transpose() << std::endl;
		//std::cout << "sqrtDDP_y_i_on_fi: \n" << sqrtDDP_y_i_on_fi.transpose() << std::endl;

		DiagMatrixXd W(m_ddLogPi);
		DiagMatrixXd WSqrt(m_sqrtDDLogPi);
		const Eigen::MatrixXd C = eye + (WSqrt * K * WSqrt);
		// get cholesky from C
		m_lowerOfCholesky = Eigen::LLT<Eigen::MatrixXd>(C).matrixL();
		const Eigen::VectorXd b = W * m_f + m_dLogPi;
		const Eigen::MatrixXd nenner = m_lowerOfCholesky.triangularView<Eigen::Lower>().solve(WSqrt * K * b);
		a = b - WSqrt * m_lowerOfCholesky.transpose().triangularView<Eigen::Upper>().solve(nenner);
		//std::cout << "a: \n" << a.transpose() << std::endl;
		m_f = K * a;
		const double objective = -0.5 * (double) (a.transpose() * m_f);
		std::cout << "\rError in " << j <<": " << fabs(lastObjective / objective - 1.0) << ", from: " << lastObjective << ", to: " << objective <<  "                    ";
		flush(std::cout);
		converged = fabs(lastObjective / objective - 1.0) < 0.01;
		lastObjective = objective;
		//converged = m_f.mean() < 100 && m_f.mean() > 50;//fabs((m_f-lastF).mean()) < 0.0001;
		++j;
	}
	// marginal likelihood?
	/*Eigen::VectorXd minusLogPi;
	for(int i = 0; i < dataPoints; ++i){
		minusLogPi[i] = -log(1.0 / (1.0 + exp((double) -y[i] * (double) f[i]))); // check if this is the probability!
	}
	Eigen::VectorXd logQ = -0.5 * (a.transpose() * f) + minusLogPi -
	*/
}

double GaussianProcessBinaryClass::predict(const Eigen::VectorXd newPoint){
	const DiagMatrixXd WSqrt(m_sqrtDDLogPi);
	Eigen::VectorXd kXStar;
	GaussianProcessMultiClass::kernelVector(newPoint, m_dataMat, kXStar);
	const double fStar = (double) (kXStar.transpose() * (m_dLogPi));
	const Eigen::VectorXd v = m_lowerOfCholesky.triangularView<Eigen::Lower>().solve(WSqrt * kXStar);
	const double vFStar = fabs((GaussianProcessMultiClass::m_sigmaN * GaussianProcessMultiClass::m_sigmaN + 1) - v.transpose() * v);

	std::cout << "fStar: " << fStar << ", vFStar: " << vFStar << std::endl;

	const int amountOfSamples = 10000;
	const double start = fStar - vFStar * 3;
	const double end = fStar + vFStar * 3;
	const double stepSize = (end- start) / amountOfSamples;
	double prob = 0;
	for(double p = start; p < end; p+=stepSize){
		const double x = rand()/static_cast<double>(RAND_MAX);
		const double y = rand()/static_cast<double>(RAND_MAX);
		const double result = cos(2.0 * M_PI * x)*sqrt(-2*log(y)); // random gaussian after box mueller
		const double height = 1.0 / (1.0 + exp(p)) * (result * vFStar + fStar);
		prob += height * stepSize; // gives the integral
	}
	return prob;
}
