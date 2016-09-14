/*
 * Kernel.cc
 *
 *  Created on: 01.07.2016
 *      Author: Max
 */

#include "Kernel.h"
#include <cmath>

Kernel::Kernel(): m_init(false), m_dataPoints(0), m_randLenMean(0), m_randLenVar(0){
	m_hyperParams[0] = m_hyperParams[1] = m_hyperParams[2] = 0.0;
}

Kernel::~Kernel(){
}

void Kernel::init(const Eigen::MatrixXd& dataMat){
	m_dataPoints = dataMat.cols();
	m_differences = Eigen::MatrixXd(m_dataPoints , m_dataPoints );
	int counter = 0;
	m_randLenMean = 0;
	for(int i = 0; i < m_dataPoints ; ++i){
		for(int j = i + 1; j < m_dataPoints ; ++j){
			const Eigen::VectorXd diff = dataMat.col(i) - dataMat.col(j);
			m_differences(i,j) = diff.squaredNorm();
			m_differences(j,i) = m_differences(i,j);
			const double frac = (double) (counter) / (double) (counter + 1) ;
			m_randLenMean = m_randLenMean * frac + (double) m_differences(i,j) * (1.0-frac);
			++counter;
		}
	}
	counter = 0;
	m_randLenVar = 0;
	for(int i = 0; i < m_dataPoints ; ++i){
		for(int j = i + 1; j < m_dataPoints ; ++j){
			const double update = (m_randLenMean - (double) m_differences(i,j));
			const double frac = (double) (counter) / (double)  (counter + 1) ;
			m_randLenVar = m_randLenVar * frac + update * update * (1.0-frac);
			++counter;
		}
	}
	m_randLenVar = sqrt(m_randLenVar);
	//std::cout << "Rand mean is: " << m_randLenMean << std::endl;
	//std::cout << "Rand var is:  " << m_randLenVar << std::endl;
	m_randLen.reset(m_randLenMean, m_randLenVar);
	m_randSigmaF.reset(1.0, 0.25);
	m_init = true;
}

double Kernel::kernelFunc(const int row, const int col) const{
	return m_hyperParams[1] * m_hyperParams[1] * exp(-0.5 * (1.0/ (m_hyperParams[0] * m_hyperParams[0])) * (double) m_differences(row, col));
}

double Kernel::kernelFuncVec(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const{
	return m_hyperParams[1] * m_hyperParams[1] * exp(-0.5 * (1.0/ (m_hyperParams[0] * m_hyperParams[0])) * (double) (lhs-rhs).squaredNorm());
}

double Kernel::kernelFuncDerivativeToLength(const int row, const int col) const{
	const double lenSquared = m_hyperParams[0] * m_hyperParams[0];
	const double dotResult = (double) m_differences(row, col);
	return m_hyperParams[1] * m_hyperParams[1] * exp(-0.5 * (1.0/ (lenSquared)) * dotResult) * dotResult / (lenSquared * m_hyperParams[0]); // derivative to m_hyperParams[0]
}

double Kernel::kernelFuncDerivativeToFNoise(const int row, const int col) const{
	return 2.0 * m_hyperParams[1] * exp(-0.5 * (1.0/ (m_hyperParams[0] * m_hyperParams[0])) * (double) m_differences(row, col)); // derivative to m_hyperParams[1]
}

void Kernel::calcCovariance(Eigen::MatrixXd& cov) const{
	if(!m_init){
		printError("Kernel not inited!");
		return;
	}
	const double sigmaNSquared = m_hyperParams[2] * m_hyperParams[2];
	cov.conservativeResize(m_dataPoints, m_dataPoints);
	for(int i = 0; i < m_dataPoints; ++i){
		cov(i,i) = sigmaNSquared;
		for(int j = i + 1; j < m_dataPoints; ++j){
			cov(i,j) = kernelFunc(i,j);
			cov(j,i) = cov(i,j);
		}
	}
}

void Kernel::calcCovarianceDerivative(Eigen::MatrixXd& cov, const ParamType type) const{
	cov = Eigen::MatrixXd::Identity(m_dataPoints, m_dataPoints);
	switch(type){
		case ParamType::LENGTH:{
			const double sigmaNSquared = m_hyperParams[2] * m_hyperParams[2];
			for(int i = 0; i < m_dataPoints; ++i){
				cov(i,i) = cov.col(i)[i] * sigmaNSquared;
				for(int j = i + 1; j < m_dataPoints; ++j){
					cov(i,j) = (double) kernelFuncDerivativeToLength(i,j);
					cov(j,i) = cov(i,j);
				}
			}
			break;
		} case ParamType::FNOISE:{
			const double sigmaNSquared = m_hyperParams[2] * m_hyperParams[2];
			for(int i = 0; i < m_dataPoints; ++i){
				cov(i,i) = cov.col(i)[i] * sigmaNSquared;
				for(int j = i + 1; j < m_dataPoints; ++j){
					cov(i,j) = (double) kernelFuncDerivativeToFNoise(i,j);
					cov(j,i) = cov(i,j);
				}
			}
			break;
		} case ParamType::NNOISE:{
			for(int i = 0; i < m_dataPoints; ++i){
				cov(i,i) = 2 * m_hyperParams[2]; // derivative of m_hyperParams[2]^2
			}
			break;
		}
	}
}

void Kernel::calcKernelVector(const Eigen::VectorXd& vector, const Eigen::MatrixXd& dataMat, Eigen::VectorXd& res) const{
	res = Eigen::VectorXd(m_dataPoints);
	for(int i = 0; i < m_dataPoints; ++i){
		res[i] = (double) kernelFuncVec(vector, dataMat.col(i));
	}
}
