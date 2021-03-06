/*
 * GaussianKernel.cc
 *
 *  Created on: 01.07.2016
 *      Author: Max
 */

#include "GaussianKernel.h"

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
void GaussianKernel::completeInit(const Matrix& dataMat){
	m_dataPoints = dataMat.cols();
	m_differences = Matrix(m_dataPoints, m_dataPoints);
	int counter = 0;
	m_randLenMean = 0;
	for(int i = 0; i < m_dataPoints ; ++i){
		for(int j = i + 1; j < m_dataPoints ; ++j){
			const VectorX diff = dataMat.col(i) - dataMat.col(j);
			m_differences(i,j) = diff.squaredNorm();
			m_differences(j,i) = m_differences(i,j);
			const Real frac = (Real) (counter) / (Real) (counter + 1) ;
			m_randLenMean = m_randLenMean * frac + (Real) m_differences(i,j) * (1.0-frac);
			++counter;
		}
	}
	counter = 0;
	m_randLenVar = 0;
	for(int i = 0; i < m_dataPoints ; ++i){
		for(int j = i + 1; j < m_dataPoints ; ++j){
			const Real update = (m_randLenMean - (Real) m_differences(i,j));
			const Real frac = (Real) (counter) / (Real)  (counter + 1) ;
			m_randLenVar = m_randLenVar * frac + update * update * (1.0-frac);
			++counter;
		}
	}
	m_randLenVar = sqrtReal(m_randLenVar);
	//std::cout << "Rand mean is: " << m_randLenMean << std::endl;
	//std::cout << "Rand var is:  " << m_randLenVar << std::endl;
	m_randLen.reset(m_randLenMean, m_randLenVar);
	m_randSigmaF.reset(1.0, 0.25);
	m_init = true;
}*/

Real GaussianKernel::kernelFunc(const int row, const int col) const{
	if(!m_calcedDifferenceMatrix){
		if(m_init){
			if(m_pData != nullptr){
				VectorX* lhs = (*m_pData)[row];
				VectorX* rhs = (*m_pData)[col];
				return kernelFuncVec(*lhs, *rhs);
			}else if(m_pDataMat != nullptr){
				return kernelFuncVec(m_pDataMat->col(row), m_pDataMat->col(row));
			}
		}else{
			printError("The init process failed, init was tried: " << m_init);
		}
	}else{
		return m_kernelParams.m_fNoise.getSquaredValue() * expReal(-0.5 * m_kernelParams.m_length.getSquaredInverseValue()
				* (Real) (*m_differences).coeff(row, col)) + m_kernelParams.m_sNoise.getSquaredValue();
	}
	return 0;
}

Real GaussianKernel::kernelFuncVec(const VectorX& lhs, const VectorX& rhs) const {
	if(hasLengthMoreThanOneDim()){
		Real squaredNorm = 0;
		for(unsigned int i = 0; i < lhs.rows(); ++i){
			const Real temp = (lhs.coeff(i) - rhs.coeff(i)) * 1. / m_kernelParams.m_length.getValues()[i];
			squaredNorm += temp * temp;
		}
		return m_kernelParams.m_fNoise.getValue() * m_kernelParams.m_fNoise.getValue() *
				expReal(-0.5 * squaredNorm) + m_kernelParams.m_sNoise.getValue() * m_kernelParams.m_sNoise.getValue();
	}else{
		return m_kernelParams.m_fNoise.getSquaredValue() * expReal(-0.5 * m_kernelParams.m_length.getSquaredInverseValue()
				* (Real) (lhs-rhs).squaredNorm()) + m_kernelParams.m_sNoise.getSquaredValue();
	}
	return 0;
}

Real GaussianKernel::kernelFuncDerivativeToParam(const int row, const int col, const OwnKernelElement* type, const int element) const {
	if(type->getKernelNr() == m_kernelParams.m_length.getKernelNr()){
		if(hasLengthMoreThanOneDim() && element != -1){
			Real squaredNorm = 0;
			const Real lenElement = m_kernelParams.m_length.getValues()[element];
			Real squaredDerivNorm = 0;
			if(m_pData != nullptr){
				for(unsigned int i = 0; i < (*m_pData)[row]->rows(); ++i){
					const Real temp = ((*(*m_pData)[row]).coeff(i) - (*(*m_pData)[col]).coeff(i)) * 1. / m_kernelParams.m_length.getValues()[i];
					squaredNorm += temp * temp;
				}
				squaredDerivNorm = ((*(*m_pData)[row]).coeff(element) - (*(*m_pData)[col]).coeff(element)) * ((*(*m_pData)[row]).coeff(element) - (*(*m_pData)[col]).coeff(element)) /
						(lenElement * lenElement * lenElement);
			}else if(m_pDataMat != nullptr){
				for(unsigned int i = 0; i < m_pDataMat->col(row).rows(); ++i){
					const Real temp = (m_pDataMat->col(row).coeff(i) - m_pDataMat->col(col).coeff(i)) * 1. / m_kernelParams.m_length.getValues()[i];
					squaredNorm += temp * temp;
				}
				squaredDerivNorm = (m_pDataMat->col(row).coeff(element) - m_pDataMat->col(col).coeff(element)) * (m_pDataMat->col(row).coeff(element) - m_pDataMat->col(col).coeff(element)) /
						(lenElement * lenElement * lenElement);
			}
			return m_kernelParams.m_fNoise.getSquaredValue() * expReal(-0.5 * squaredNorm) * squaredDerivNorm;
		}else if(m_calcedDifferenceMatrix){
			const Real lenSquared = m_kernelParams.m_length.getSquaredValue();
			const Real dotResult = (Real) (*m_differences).coeff(row, col);
			return m_kernelParams.m_fNoise.getSquaredValue() * expReal(-0.5 * m_kernelParams.m_length.getSquaredInverseValue() * dotResult)
					* dotResult / (lenSquared * m_kernelParams.m_length.getValue()); // derivative to m_params[0]
		}else{
			const Real lenSquared = m_kernelParams.m_length.getSquaredValue();
			Real dotResult = 0;
			if(m_pData != nullptr){
				VectorX* lhs = (*m_pData)[row];
				VectorX* rhs = (*m_pData)[col];
				return (*lhs - *rhs).squaredNorm();
			}else if(m_pDataMat != nullptr){
				return (m_pDataMat->col(row) - m_pDataMat->col(row)).squaredNorm();
			}
			return m_kernelParams.m_fNoise.getSquaredValue() * expReal(-0.5 * m_kernelParams.m_length.getSquaredInverseValue() * dotResult)
					* dotResult / (lenSquared * m_kernelParams.m_length.getValue());
		}
	}else if(type->getKernelNr() == m_kernelParams.m_fNoise.getKernelNr()){
		if(!hasLengthMoreThanOneDim() && m_calcedDifferenceMatrix){
			return 2.0 * m_kernelParams.m_fNoise.getValue() * expReal(-0.5 * (1.0/ (m_kernelParams.m_length.getValue() *
					m_kernelParams.m_length.getValue())) * (Real) m_differences->coeff(row, col)); // derivative to m_params[1]
		}else{
			Real result = 0;
			if(m_pData != nullptr){
				VectorX* lhs = (*m_pData)[row];
				VectorX* rhs = (*m_pData)[col];
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
Real GaussianKernel::kernelFuncDerivativeToLength(const int row, const int col) const{
	const Real lenSquared = m_kernelParams.m_length.getValue() * m_kernelParams.m_length.getValue();
	const Real dotResult = (Real) m_differences(row, col);
	return m_kernelParams.m_fNoise.getValue() * m_kernelParams.m_fNoise.getValue() * expReal(-0.5 * (1.0/ (lenSquared)) * dotResult) * dotResult / (lenSquared * m_kernelParams.m_length.getValue()); // derivative to m_params[0]
}

Real GaussianKernel::kernelFuncDerivativeToFNoise(const int row, const int col) const{
	return 2.0 * m_kernelParams.m_fNoise.getValue() * expReal(-0.5 * (1.0/ (m_kernelParams.m_length.getValue() *
			m_kernelParams.m_length.getValue())) * (Real) m_differences(row, col)); // derivative to m_params[1]
}
*/

void GaussianKernel::calcCovariance(Matrix& cov) const{
	cov = Matrix(m_dataPoints, m_dataPoints);
	for(unsigned int i = 0; i < m_dataPoints; ++i){
		cov.coeffRef(i,i) = calcDiagElement(i);
		for(unsigned int j = i + 1; j < m_dataPoints; ++j){
			cov.coeffRef(i,j) = kernelFunc(i, j);
			cov.coeffRef(j,i) = cov.coeff(i,j);
		}
	}
}

void GaussianKernel::calcCovarianceDerivativeForInducingPoints(Matrix& cov, const std::list<unsigned int>& activeSet, const OwnKernelElement* type, const int element) const{
	const int nrOfInducingPoints = activeSet.size();
	if(!type->isDerivativeOnlyDiag()){
		cov.resize(nrOfInducingPoints, nrOfInducingPoints);
		unsigned int i = 0;
		for(auto it1 = activeSet.cbegin(); it1 != activeSet.cend(); ++it1, ++i){
			cov.coeffRef(i,i) = calcDerivativeDiagElement(*it1, type);
			unsigned int j = i + 1;
			auto it2 = it1;
			++it2;
			for(; it2 != activeSet.end(); ++it2, ++j){
				cov.coeffRef(i,j) = kernelFuncDerivativeToParam(*it1, *it2, type, element);
				cov.coeffRef(j,i) = cov.coeff(i,j);
			}
		}
	}else{
		cov = Matrix::Zero(nrOfInducingPoints, nrOfInducingPoints);
		for(int i = 0; i < nrOfInducingPoints; ++i){
			cov.coeffRef(i,i) = calcDerivativeDiagElement(i, type); // derivative of m_params[2]^2
		}
	}
}

Real GaussianKernel::calcDerivativeDiagElement(unsigned int row, const OwnKernelElement* type) const{
	UNUSED(row);
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

void GaussianKernel::calcCovarianceDerivative(Matrix& cov, const OwnKernelElement* type) const{
	cov = Matrix(m_dataPoints, m_dataPoints);
	for(unsigned int i = 0; i < m_dataPoints; ++i){
		cov.coeffRef(i,i) = calcDerivativeDiagElement(i, type);
		for(unsigned int j = i + 1; j < m_dataPoints; ++j){
			cov.coeffRef(i,j) = kernelFuncDerivativeToParam(i, j, type);
			cov.coeffRef(j,i) = cov.coeff(i,j);
		}
	}
}

void GaussianKernel::calcKernelVector(const VectorX& vector, const Matrix& dataMat, VectorX& res) const{
	res = VectorX(m_dataPoints);
	for(unsigned int i = 0; i < m_dataPoints; ++i){
		res.coeffRef(i) = (Real) kernelFuncVec(vector, dataMat.col(i));
	}
}

std::string GaussianKernel::prettyString() const{
	std::stringstream ss;
	ss << m_kernelParams;
	return ss.str();
}

void GaussianKernel::setHyperParams(Real len, Real noiseF){
	m_kernelParams.m_length.setAllValuesTo(len);
	m_kernelParams.m_fNoise.setAllValuesTo(noiseF);
}

void GaussianKernel::setHyperParams(Real len, Real noiseF, Real noiseS){
	m_kernelParams.m_length.setAllValuesTo(len);
	m_kernelParams.m_fNoise.setAllValuesTo(noiseF);
	m_kernelParams.m_sNoise.setAllValuesTo(noiseS);
}

void GaussianKernel::setHyperParams(const std::vector<Real>& len, Real noiseF, Real noiseS){
	m_kernelParams.m_length.changeAmountOfDims(true);
	for(unsigned int i = 0; i < ClassKnowledge::instance().amountOfDims(); ++i){
		m_kernelParams.m_length.getValues()[i] = len[i];
	}
	m_kernelParams.m_fNoise.setAllValuesTo(noiseF);
	m_kernelParams.m_sNoise.setAllValuesTo(noiseS);
}

void GaussianKernel::addHyperParams(Real len, Real noiseF, Real noiseS){
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
		for(unsigned int i = 0; i < ClassKnowledge::instance().amountOfDims(); ++i){
			params.m_length.getValues()[i] = m_kernelParams.m_length.getValues()[i];
		}
	}else{
		params.m_length.getValues()[0] = m_kernelParams.m_length.getValue();
	}
}

