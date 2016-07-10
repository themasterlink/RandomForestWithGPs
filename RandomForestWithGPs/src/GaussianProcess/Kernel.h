/*
 * Kernel.h
 *
 *  Created on: 01.07.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_KERNEL_H_
#define GAUSSIANPROCESS_KERNEL_H_

#include "../Data/Data.h"
#include "../RandomNumberGenerator/RandomGaussianNr.h"

class Kernel{
public:

	enum ParamType { // same order as in m_hyperParams!
		LENGTH,
		FNOISE,
		NNOISE
	};

	Kernel();
	virtual ~Kernel();

	void init(const Eigen::MatrixXd& dataMat);

	void calcCovariance(Eigen::MatrixXd& cov) const;

	void calcCovarianceDerivative(Eigen::MatrixXd& cov, const ParamType type) const;

	void calcKernelVector(const Eigen::VectorXd& vector, const Eigen::MatrixXd& dataMat, Eigen::VectorXd& res) const;

	void addHyperParams(const double len, const double sigmaF, const double sigmaN);

	void addHyperParams(const std::vector<double>& values);

	void subHyperParams(const std::vector<double>& values);

	void setHyperParams(const double len, const double sigmaF, const double sigmaN);

	void setHyperParams(const std::vector<double>& values);

	void getHyperParams(std::vector<double>& values) const;

	void newRandHyperParams();

	double len() const {    return m_hyperParams[0];};
	double sigmaF() const { return m_hyperParams[1];};
	double sigmaN() const { return m_hyperParams[2];};

	double getLenMean() const {return m_randLenMean;};

	double getLenVar() const {return m_randLenVar;};

	std::string prettyString() const {
		std::stringstream ss;
		ss << "len: " << m_hyperParams[0] << ", sigmaF: " << m_hyperParams[1] << ", sigmaN: " << m_hyperParams[2];
		return ss.str();
	}

private:
	double kernelFunc(const int row, const int col) const;

	double kernelFuncVec(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const;

	double kernelFuncDerivativeToLength(const int row, const int col) const;

	double kernelFuncDerivativeToFNoise(const int row, const int col) const;

	Eigen::MatrixXd m_differences;
	bool m_init;
	int m_dataPoints;
	double m_randLenMean;
	double m_randLenVar;
	double m_hyperParams[3]; // order is len, sigmaF, sigmaN

	RandomGaussianNr m_randLen;
	RandomGaussianNr m_randSigmaF;
};

inline
void Kernel::addHyperParams(const double len, const double sigmaF, const double sigmaN){
	m_hyperParams[0] += len;
	m_hyperParams[1] += sigmaF;
	m_hyperParams[2] += sigmaN;
}

inline
void Kernel::setHyperParams(const double len, const double sigmaF, const double sigmaN){
	m_hyperParams[0] = len;
	m_hyperParams[1] = sigmaF;
	m_hyperParams[2] = sigmaN;
}

inline
void Kernel::setHyperParams(const std::vector<double>& values){
	m_hyperParams[0] = values[0];
	m_hyperParams[1] = values[1];
	m_hyperParams[2] = values[2];
}

inline
void Kernel::addHyperParams(const std::vector<double>& values){
	m_hyperParams[0] += values[0];
	m_hyperParams[1] += values[1];
	m_hyperParams[2] += values[2];
}

inline
void Kernel::subHyperParams(const std::vector<double>& values){
	m_hyperParams[0] -= values[0];
	m_hyperParams[1] -= values[1];
	m_hyperParams[2] -= values[2];
}

inline
void Kernel::getHyperParams(std::vector<double>& values) const{
	values[0] = m_hyperParams[0];
	values[1] = m_hyperParams[1];
	values[2] = m_hyperParams[2];
}

inline
void Kernel::newRandHyperParams(){
	m_hyperParams[0] = m_randLen();
	m_hyperParams[1] = m_randSigmaF();
	m_hyperParams[2] = ((double) rand() / (RAND_MAX)) + 0.1;
}

#endif /* GAUSSIANPROCESS_KERNEL_H_ */
