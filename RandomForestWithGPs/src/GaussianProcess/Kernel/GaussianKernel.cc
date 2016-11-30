/*
 * GaussianKernel.cc
 *
 *  Created on: 01.07.2016
 *      Author: Max
 */

#include "GaussianKernel.h"
#include "../../Utility/ReadWriterHelper.h"
#include <cmath>
#include "../../Base/Settings.h"

GaussianKernel::GaussianKernel(bool simpleLength): KernelBase<GaussianKernelParams>(GaussianKernelInitParams(simpleLength)){
	for(unsigned int i = 0; i < getNrOfParams(); ++i){
		m_kernelParams.m_params[i]->setAllValuesTo(0);
	}
}

void GaussianKernel::changeKernelConfig(const bool useAllDimForLen){
	m_kernelParams.m_length.changeAmountOfDims(useAllDimForLen);
}

GaussianKernel::~GaussianKernel(){
}
/*
void GaussianKernel::completeInit(const Eigen::MatrixXd& dataMat){
	m_dataPoints = dataMat.cols();
	m_differences = Eigen::MatrixXd(m_dataPoints, m_dataPoints);
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
}*/

double GaussianKernel::kernelFunc(const int row, const int col) const{
	if(!m_calcedDifferenceMatrix){
		if(m_init){
			if(m_pData != nullptr){
				Eigen::VectorXd* lhs = (*m_pData)[row];
				Eigen::VectorXd* rhs = (*m_pData)[col];
				return kernelFuncVec(*lhs, *rhs);
			}else if(m_pDataMat != nullptr){
				return kernelFuncVec(m_pDataMat->col(row), m_pDataMat->col(row));
			}
		}else{
			printError("The init process failed, init was tried: " << m_init);
		}
	}else{
		return m_kernelParams.m_fNoise.getSquaredValue() * exp(-0.5 * m_kernelParams.m_length.getSquaredInverseValue()
				* (double) (*m_differences)(row, col)) + m_kernelParams.m_sNoise.getSquaredValue();
	}
	return 0;
}

double GaussianKernel::kernelFuncVec(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
	if(hasLengthMoreThanOneDim()){
		double squaredNorm = 0;
		for(unsigned int i = 0; i < lhs.rows(); ++i){
			const double temp = (lhs[i] - rhs[i]) * 1. / m_kernelParams.m_length.getValues()[i];
			squaredNorm += temp * temp;
		}
		return m_kernelParams.m_fNoise.getValue() * m_kernelParams.m_fNoise.getValue() *
				exp(-0.5 * squaredNorm) + m_kernelParams.m_sNoise.getValue() * m_kernelParams.m_sNoise.getValue();
	}else{
		return m_kernelParams.m_fNoise.getSquaredValue() * exp(-0.5 * m_kernelParams.m_length.getSquaredInverseValue()
				* (double) (lhs-rhs).squaredNorm()) + m_kernelParams.m_sNoise.getSquaredValue();
	}
	return 0;
}

double GaussianKernel::kernelFuncDerivativeToParam(const int row, const int col, const OwnKernelElement* type, const int element) const {
	if(type->getKernelNr() == m_kernelParams.m_length.getKernelNr()){
		if(hasLengthMoreThanOneDim() && element != -1){
			double squaredNorm = 0;
			const double lenElement = m_kernelParams.m_length.getValues()[element];
			double squaredDerivNorm = 0;
			if(m_pData != nullptr){
				for(unsigned int i = 0; i < (*m_pData)[row]->rows(); ++i){
					const double temp = ((*(*m_pData)[row])[i] - (*(*m_pData)[col])[i]) * 1. / m_kernelParams.m_length.getValues()[i];
					squaredNorm += temp * temp;
				}
				squaredDerivNorm = ((*(*m_pData)[row])[element] - (*(*m_pData)[col])[element]) * ((*(*m_pData)[row])[element] - (*(*m_pData)[col])[element]) /
						(lenElement * lenElement * lenElement);
			}else if(m_pDataMat != nullptr){
				for(unsigned int i = 0; i < m_pDataMat->col(row).rows(); ++i){
					const double temp = (m_pDataMat->col(row)[i] - m_pDataMat->col(col)[i]) * 1. / m_kernelParams.m_length.getValues()[i];
					squaredNorm += temp * temp;
				}
				squaredDerivNorm = (m_pDataMat->col(row)[element] - m_pDataMat->col(col)[element]) * (m_pDataMat->col(row)[element] - m_pDataMat->col(col)[element]) /
						(lenElement * lenElement * lenElement);
			}
			return m_kernelParams.m_fNoise.getSquaredValue() * exp(-0.5 * squaredNorm) * squaredDerivNorm;
		}else if(m_calcedDifferenceMatrix){
			const double lenSquared = m_kernelParams.m_length.getSquaredValue();
			const double dotResult = (double) (*m_differences)(row, col);
			return m_kernelParams.m_fNoise.getSquaredValue() * exp(-0.5 * m_kernelParams.m_length.getSquaredInverseValue() * dotResult)
					* dotResult / (lenSquared * m_kernelParams.m_length.getValue()); // derivative to m_params[0]
		}else{
			const double lenSquared = m_kernelParams.m_length.getSquaredValue();
			double dotResult = 0;
			if(m_pData != nullptr){
				Eigen::VectorXd* lhs = (*m_pData)[row];
				Eigen::VectorXd* rhs = (*m_pData)[col];
				return (*lhs - *rhs).squaredNorm();
			}else if(m_pDataMat != nullptr){
				return (m_pDataMat->col(row) - m_pDataMat->col(row)).squaredNorm();
			}
			return m_kernelParams.m_fNoise.getSquaredValue() * exp(-0.5 * m_kernelParams.m_length.getSquaredInverseValue() * dotResult)
					* dotResult / (lenSquared * m_kernelParams.m_length.getValue());
		}
	}else if(type->getKernelNr() == m_kernelParams.m_fNoise.getKernelNr()){
		if(!hasLengthMoreThanOneDim() && m_calcedDifferenceMatrix){
			return 2.0 * m_kernelParams.m_fNoise.getValue() * exp(-0.5 * (1.0/ (m_kernelParams.m_length.getValue() *
					m_kernelParams.m_length.getValue())) * (double) (*m_differences)(row, col)); // derivative to m_params[1]
		}else{
			double result = 0;
			if(m_pData != nullptr){
				Eigen::VectorXd* lhs = (*m_pData)[row];
				Eigen::VectorXd* rhs = (*m_pData)[col];
				result = kernelFuncVec(*lhs, *rhs);
			}else if(m_pDataMat != nullptr){
				result = kernelFuncVec(m_pDataMat->col(row), m_pDataMat->col(row));
			}
			// subtract the sNoise divide through one fNoise and multiply with 2.0, is the easiest way to get the derivative
			return (result - m_kernelParams.m_sNoise.getSquaredValue()) / m_kernelParams.m_fNoise.getValue() * 2.0;
		}
	}else if(type->getKernelNr() == m_kernelParams.m_sNoise.getKernelNr()){
		return 2.0 * m_kernelParams.m_sNoise.getValue();
	}else{
		if(m_calcedDifferenceMatrix){
			printError("This kernel type is not supported!");
		}else{
			printError("This kernel type is not supported and there is only one length param but, the difference matrix was not calculated!");
		}
	}
	return 0.;
}

/*
double GaussianKernel::kernelFuncDerivativeToLength(const int row, const int col) const{
	const double lenSquared = m_kernelParams.m_length.getValue() * m_kernelParams.m_length.getValue();
	const double dotResult = (double) m_differences(row, col);
	return m_kernelParams.m_fNoise.getValue() * m_kernelParams.m_fNoise.getValue() * exp(-0.5 * (1.0/ (lenSquared)) * dotResult) * dotResult / (lenSquared * m_kernelParams.m_length.getValue()); // derivative to m_params[0]
}

double GaussianKernel::kernelFuncDerivativeToFNoise(const int row, const int col) const{
	return 2.0 * m_kernelParams.m_fNoise.getValue() * exp(-0.5 * (1.0/ (m_kernelParams.m_length.getValue() *
			m_kernelParams.m_length.getValue())) * (double) m_differences(row, col)); // derivative to m_params[1]
}
*/

void GaussianKernel::calcCovariance(Eigen::MatrixXd& cov) const{
	cov = Eigen::MatrixXd(m_dataPoints, m_dataPoints);
	for(int i = 0; i < m_dataPoints; ++i){
		cov(i,i) = calcDiagElement(i);
		for(int j = i + 1; j < m_dataPoints; ++j){
			cov(i,j) = kernelFunc(i, j);
			cov(j,i) = cov(i,j);
		}
	}
}

void GaussianKernel::calcCovarianceDerivativeForInducingPoints(Eigen::MatrixXd& cov, const std::list<int>& activeSet, const OwnKernelElement* type, const int element) const{
	const int nrOfInducingPoints = activeSet.size();
	if(!type->isDerivativeOnlyDiag()){
		cov.resize(nrOfInducingPoints, nrOfInducingPoints);
		unsigned int i = 0;
		for(std::list<int>::const_iterator it1 = activeSet.begin(); it1 != activeSet.end(); ++it1, ++i){
			cov(i,i) = calcDerivativeDiagElement(*it1, type);
			unsigned int j = i + 1;
			std::list<int>::const_iterator it2 = it1;
			++it2;
			for(; it2 != activeSet.end(); ++it2, ++j){
				cov(i,j) = kernelFuncDerivativeToParam(*it1, *it2, type, element);
				cov(j,i) = cov(i,j);
			}
		}
	}else{
		cov = Eigen::MatrixXd::Zero(nrOfInducingPoints, nrOfInducingPoints);
		for(int i = 0; i < nrOfInducingPoints; ++i){
			cov(i,i) = calcDerivativeDiagElement(i, type); // derivative of m_params[2]^2
		}
	}
}

double GaussianKernel::calcDerivativeDiagElement(unsigned int row, const OwnKernelElement* type) const{
	if(type->getKernelNr() == m_kernelParams.m_length.getKernelNr()){
		return m_kernelParams.m_fNoise.getSquaredValue();
	}else if(type->getKernelNr() == m_kernelParams.m_fNoise.getKernelNr()){
		return 2.0 * m_kernelParams.m_fNoise.getValue();
	}else if(type->getKernelNr() == m_kernelParams.m_sNoise.getKernelNr()){
		return 2.0 * m_kernelParams.m_sNoise.getValue();
	}else{
		printError("This type is not defined!");
		return 0;
	}
}

void GaussianKernel::calcCovarianceDerivative(Eigen::MatrixXd& cov, const OwnKernelElement* type) const{
	cov = Eigen::MatrixXd(m_dataPoints, m_dataPoints);
	for(int i = 0; i < m_dataPoints; ++i){
		cov(i,i) = calcDerivativeDiagElement(i, type);
		for(int j = i + 1; j < m_dataPoints; ++j){
			cov(i,j) = kernelFuncDerivativeToParam(i, j, type);
			cov(j,i) = cov(i,j);
		}
	}
}

void GaussianKernel::calcKernelVector(const Eigen::VectorXd& vector, const Eigen::MatrixXd& dataMat, Eigen::VectorXd& res) const{
	res = Eigen::VectorXd(m_dataPoints);
	for(int i = 0; i < m_dataPoints; ++i){
		res[i] = (double) kernelFuncVec(vector, dataMat.col(i));
	}
}

std::string GaussianKernel::prettyString() const{
	std::stringstream ss;
	ss << m_kernelParams;
	return ss.str();
}

void GaussianKernel::setHyperParams(double len, double noiseF){
	m_kernelParams.m_length.setAllValuesTo(len);
	m_kernelParams.m_fNoise.setAllValuesTo(noiseF);
}

void GaussianKernel::setHyperParams(double len, double noiseF, double noiseS){
	m_kernelParams.m_length.setAllValuesTo(len);
	m_kernelParams.m_fNoise.setAllValuesTo(noiseF);
	m_kernelParams.m_sNoise.setAllValuesTo(noiseS);
}

void GaussianKernel::setHyperParams(const std::vector<double>& len, double noiseF, double noiseS){
	m_kernelParams.m_length.changeAmountOfDims(true);
	for(unsigned int i = 0; i < ClassKnowledge::amountOfDims(); ++i){
		m_kernelParams.m_length.getValues()[i] = len[i];
	}
	m_kernelParams.m_fNoise.setAllValuesTo(noiseF);
	m_kernelParams.m_sNoise.setAllValuesTo(noiseS);
}

void GaussianKernel::addHyperParams(double len, double noiseF, double noiseS){
	if(hasLengthMoreThanOneDim()){
		printError("This function should not be called if there is more than one parameter for the length scale!");
	}else{
		m_kernelParams.m_length.addToFirstValue(len);
		m_kernelParams.m_fNoise.addToFirstValue(noiseF);
		m_kernelParams.m_sNoise.addToFirstValue(noiseS);
	}
}

void GaussianKernel::getCopyOfParams(GaussianKernelParams& params){
	params.m_fNoise.setAllValuesTo(m_kernelParams.m_fNoise.getValue());
	params.m_sNoise.setAllValuesTo(m_kernelParams.m_sNoise.getValue());
	params.m_length.changeAmountOfDims(hasLengthMoreThanOneDim());
	if(hasLengthMoreThanOneDim()){
		for(unsigned int i = 0; i < ClassKnowledge::amountOfDims(); ++i){
			params.m_length.getValues()[i] = m_kernelParams.m_length.getValues()[i];
		}
	}else{
		params.m_length.getValues()[0] = m_kernelParams.m_length.getValue();
	}
}

