/*
 * KernelBase.cc
 *
 *  Created on: 31.10.2016
 *      Author: Max
 */

#ifndef __INCLUDE_KERNELBASE
#error "Don't include KernelBase_i.h directly. Include KernelBase.h instead."
#endif

template<typename KernelType, unsigned int nrOfParams>
KernelBase<KernelType, nrOfParams>::KernelBase(const OwnKernelInitParams& initParams, const bool sampleNewParams):
	m_differences(nullptr), m_pDataMat(nullptr), m_pData(nullptr), m_init(false),
	m_calcedDifferenceMatrix(false), m_dataPoints(0), m_kernelParams(initParams), m_seed(0.) {
	for(unsigned int i = 0; i < nrOfParams; ++i){
		m_randomGaussians[i] = new RandomGaussianNr(0.0, 1.0);
		m_kernelParams.m_params[i]->changeAmountOfDims(m_kernelParams.m_params[i]->hasMoreThanOneDim()); // to secure that the amount of values is there
		if(sampleNewParams){
			if(m_kernelParams.m_params[i]->hasMoreThanOneDim()){
				for(unsigned int j = 0; j < ClassKnowledge::amountOfDims(); ++j){
					m_kernelParams.m_params[i]->getValues()[j] = (*m_randomGaussians[i])();
				}
			}else{
				m_kernelParams.m_params[i]->getValues()[0] = (*m_randomGaussians[i])();
			}
		}
	}
}

template<typename KernelType, unsigned int nrOfParams>
KernelBase<KernelType, nrOfParams>::~KernelBase(){
	for(unsigned int i = 0; i < nrOfParams; ++i){
		SAVE_DELETE(m_randomGaussians[i]);
	}
}

template<typename KernelType, unsigned int nrOfParams>
void KernelBase<KernelType, nrOfParams>::init(const Eigen::MatrixXd& dataMat, const bool shouldDifferenceMatrixBeCalculated, const bool useSharedDifferenceMatrix){
	if(!kernelCanHaveDifferenceMatrix()){
		printError("This function should only be called from a kernel which supports the difference matrix!");
	}
	m_pDataMat = const_cast<Eigen::MatrixXd*>(&dataMat);
	m_calcedDifferenceMatrix = shouldDifferenceMatrixBeCalculated;
	if(m_calcedDifferenceMatrix){
		std::string path;
		Settings::getValue("Kernel.path", path);
		//const std::string path = "kernelFile_" + number2String(dataMat.rows()) + "_" + number2String(dataMat.cols()) + ".kernel";
		bool read = false;
		if(boost::filesystem::exists(path) && m_differences == nullptr){
			std::fstream input(path, std::ios::binary| std::ios::in);
			m_differences = new Eigen::MatrixXf();
			ReadWriterHelper::readMatrix(input, *m_differences);
			input.close();
			m_dataPoints = m_differences->cols();
			read = m_differences->cols() == dataMat.cols(); // else this means the kernel data does not fit the actual load data
			m_calcedDifferenceMatrix = true;
		}
		if(!read && !useSharedDifferenceMatrix){
			const int amountOfElementsInTriangluarMatrix = (m_pData->size() * m_pData->size() + m_pData->size()) / 2;
			m_differences = new Eigen::MatrixXf(m_dataPoints, m_dataPoints);
			calcDifferenceMatrix(0, amountOfElementsInTriangluarMatrix, m_differences);
			std::fstream output(path, std::ios::binary | std::ios::out);
			ReadWriterHelper::writeMatrix(output, *m_differences);
			output.close();
			m_calcedDifferenceMatrix = true;
		}
	}
	m_init = true;

}

template<typename KernelType, unsigned int nrOfParams>
void KernelBase<KernelType, nrOfParams>::init(const ClassData& data, const bool shouldDifferenceMatrixBeCalculated, const bool useSharedDifferenceMatrix){
	if(!kernelCanHaveDifferenceMatrix()){
		printError("This function should only be called from a kernel which supports the difference matrix!");
	}
	m_pData = const_cast<ClassData*>(&data);
	m_calcedDifferenceMatrix = shouldDifferenceMatrixBeCalculated;
	if(m_calcedDifferenceMatrix){
		std::string path;
		Settings::getValue("Kernel.path", path);
		//const std::string path = "kernelFile_" + number2String(dataMat.rows()) + "_" + number2String(dataMat.cols()) + ".kernel";
		bool read = false;
		if(boost::filesystem::exists(path)){
			std::fstream input(path, std::ios::binary| std::ios::in);
			ReadWriterHelper::readMatrix(input, *m_differences);
			input.close();
			m_dataPoints = m_differences->cols();
			read = m_dataPoints == (unsigned int) data.size(); // else this means the kernel data does not fit the actual load data
			m_calcedDifferenceMatrix = true;
		}
		if(!read && !useSharedDifferenceMatrix){
			const int amountOfElementsInTriangluarMatrix = (m_pData->size() * m_pData->size() + m_pData->size()) / 2;
			m_differences = new Eigen::MatrixXf();
			calcDifferenceMatrix(0, amountOfElementsInTriangluarMatrix, m_differences);
			m_calcedDifferenceMatrix = true;
			std::fstream output(path, std::ios::binary | std::ios::out);
			ReadWriterHelper::writeMatrix(output, *m_differences);
			output.close();
		}
	}
	m_init = true;
}

template<typename KernelType, unsigned int nrOfParams>
void KernelBase<KernelType, nrOfParams>::calcDifferenceMatrix(const int start, const int end, Eigen::MatrixXf* usedMatrix){
	m_differences = usedMatrix;
	if(m_pData != nullptr){
		m_dataPoints = m_pData->size();
		int counter = 0;
		if(usedMatrix->rows() != m_dataPoints || usedMatrix->cols() != m_dataPoints){
			m_differences->resize(m_dataPoints, m_dataPoints);
		}
		for(unsigned int i = 0; i < m_dataPoints; ++i){
			++counter;
			for(unsigned int j = i + 1; j < m_dataPoints; ++j){
				if(counter >= start){
					if(counter == end){
						i = m_dataPoints;
						break;
					}
					(*m_differences).coeffRef(i,j) = (*(*m_pData)[i] - *(*m_pData)[j]).squaredNorm();
					(*m_differences).coeffRef(j,i) = (*m_differences).coeff(i,j);
				}
				++counter;
			}
		}
		m_calcedDifferenceMatrix = true;
	}else if(m_pDataMat != nullptr){
		m_dataPoints = m_pDataMat->cols();
		int counter = 0;
		for(unsigned int i = 0; i < m_dataPoints; ++i){
			for(unsigned int j = i; j < m_dataPoints; ++j){
				if(counter >= start){
					if(counter == end){
						i = m_dataPoints;
						break;
					}
					if(i != j){
						(*m_differences).coeffRef(i,j) = (m_pDataMat->col(i) - m_pDataMat->col(j)).squaredNorm();
						(*m_differences).coeffRef(j,i) = (*m_differences).coeff(i,j);
					}else{
						(*m_differences).coeffRef(i,i) = 0.;
					}
				}
				++counter;
			}
		}
		m_calcedDifferenceMatrix = true;
	}else{
		printError("This will not work!");
	}
}


template<typename KernelType, unsigned int nrOfParams>
void KernelBase<KernelType, nrOfParams>::calcDifferenceMatrix(const int start, const int end, Eigen::MatrixXf& usedMatrix, const OnlineStorage<ClassPoint*>& storage, InformationPackage* package){
	const unsigned int dataPoints = storage.size();
	int counter = 0;
	if(usedMatrix.rows() != dataPoints || usedMatrix.cols() != dataPoints){
		printError("The size of the matrix is incorrect!");
		return;
	}
	for(unsigned int i = 0; i < dataPoints; ++i){
		++counter;
		for(unsigned int j = i + 1; j < dataPoints; ++j){
			if(counter >= start){
				if(package != nullptr && ((counter - start) % ((int) ((end - start) * 0.001))) == 0){
					std::stringstream str2;
					str2 << "Done: " << (counter - start) / (double) (end - start) * 100.0 << " %%" ;
					if(counter == start){
						package->printLineToScreenForThisThread(str2.str());
					}else{
						package->overwriteLastLineToScreenForThisThread(str2.str());
					}
				}
				if(counter == end){
					i = dataPoints;
					break;
				}
				usedMatrix.coeffRef(i,j) = (*storage[i] - *storage[j]).squaredNorm();
				usedMatrix.coeffRef(j,i) = usedMatrix.coeff(i,j);
			}
			++counter;
		}
	}
}

template<typename KernelType, unsigned int nrOfParams>
void KernelBase<KernelType, nrOfParams>::subGradient(const KernelType& gradient, const double factor){
	for(unsigned int i = 0; i < nrOfParams; ++i){
		if(m_kernelParams.m_params[i]->hasMoreThanOneDim() && gradient.m_params[i]->hasMoreThanOneDim()){ // both true
			for(unsigned int j = 0; j < ClassKnowledge::amountOfDims(); ++j){
				m_kernelParams.m_params[i]->getValues()[j] -= factor * gradient.m_params[i]->getValues()[j];
			}
		}else if(!m_kernelParams.m_params[i]->hasMoreThanOneDim() && !gradient.m_params[i]->hasMoreThanOneDim()){ // both false
			m_kernelParams.m_params[i]->getValues()[0] -= factor * gradient.m_params[i]->getValue();
		}else{
			printError("The gradient does not match the kernel!");
		}
	}
}

template<typename KernelType, unsigned int nrOfParams>
void KernelBase<KernelType, nrOfParams>::setHyperParamsWith(const KernelType& params){
	for(unsigned int i = 0; i < KernelType::paramsAmount; ++i){
		if(m_kernelParams.m_params[i]->hasMoreThanOneDim()){
			if(m_kernelParams.m_params[i]->hasMoreThanOneDim() == params.m_params[i]->hasMoreThanOneDim()){
				for(unsigned int j = 0; j < ClassKnowledge::amountOfDims(); ++j){
					m_kernelParams.m_params[i]->getValues()[j] = params.m_params[i]->getValues()[j];
				}
			}else{
				printWarning("Reduce the amount of hyperparams from: " << ClassKnowledge::amountOfDims() << ", to 1");
				m_kernelParams.m_params[i]->changeAmountOfDims(params.m_params[i]->hasMoreThanOneDim()); // hasMoreThanOneDim should be false
				// -> only one param
				m_kernelParams.m_params[i]->getValues()[0] = params.m_params[i]->getValue();
			}
		}else{
			if(m_kernelParams.m_params[i]->hasMoreThanOneDim() == params.m_params[i]->hasMoreThanOneDim()){
				m_kernelParams.m_params[i]->getValues()[0] = params.m_params[i]->getValue();
			}else{
				printWarning("Reduce the amount of hyperparams from: " << ClassKnowledge::amountOfDims() << ", to 1");
				m_kernelParams.m_params[i]->changeAmountOfDims(params.m_params[i]->hasMoreThanOneDim()); // hasMoreThanOneDim should be true
				for(unsigned int j = 0; j < ClassKnowledge::amountOfDims(); ++j){
					m_kernelParams.m_params[i]->getValues()[j] = params.m_params[i]->getValues()[j];
				}
			}
		}
	}
}

template<typename KernelType, unsigned int nrOfParams>
KernelType& KernelBase<KernelType, nrOfParams>::getHyperParams(){
	return m_kernelParams;
}

template<typename KernelType, unsigned int nrOfParams>
const KernelType& KernelBase<KernelType, nrOfParams>::getHyperParams() const {
	return m_kernelParams;
}

template<typename KernelType, unsigned int nrOfParams>
void KernelBase<KernelType, nrOfParams>::setGaussianRandomVariables(const std::vector<double>& means, const std::vector<double> sds){
	if(means.size() == sds.size() && means.size() == nrOfParams){
		for(unsigned int i = 0; i < nrOfParams; ++i){
			m_randomGaussians[i]->reset(means[i], sds[i]);
		}
	}else{
		printError("The amount of means and/or sds does not the match the amount of hyper parameters");
	}
}

template<typename KernelType, unsigned int nrOfParams>
void KernelBase<KernelType, nrOfParams>::calcCovariance(Eigen::MatrixXd& cov) const{
	if(m_init){
		cov.conservativeResize(m_dataPoints, m_dataPoints);
		for(int i = 0; i < m_dataPoints; ++i){
			cov.coeffRef(i,i) =  calcDiagElement(i);
			for(int j = i + 1; j < m_dataPoints; ++j){
				cov.coeffRef(i,j) = kernelFunc(i,j);
				cov.coeffRef(j,i) = cov.coeff(i,j);
			}
		}
	}else{
		printError("Kernel not inited!");
	}
}

template<typename KernelType, unsigned int nrOfParams>
void KernelBase<KernelType, nrOfParams>::calcCovarianceDerivative(Eigen::MatrixXd& cov, const OwnKernelElement* type) const{
	if(m_init){
		cov.conservativeResize(m_dataPoints, m_dataPoints);
		for(int i = 0; i < m_dataPoints; ++i){
			cov(i,i) =  calcDerivativeDiagElement(i, type);
			for(int j = i + 1; j < m_dataPoints; ++j){
				cov.coeffRef(i,j) = kernelFunc(i,j);
				cov.coeffRef(j,i) = cov.coeff(i,j);
			}
		}
	}else{
		printError("Kernel not inited!");
	}
}

template<typename KernelType, unsigned int nrOfParams>
void KernelBase<KernelType, nrOfParams>::calcCovarianceDerivativeForInducingPoints(Eigen::MatrixXd& cov, const std::list<int>& activeSet, const OwnKernelElement* type) const{
	const int nrOfInducingPoints = activeSet.size();
	cov = Eigen::MatrixXd(nrOfInducingPoints, nrOfInducingPoints);
	if(!type->isDerivativeOnlyDiag()){
		unsigned int i = 0;
		for(std::list<int>::const_iterator it1 = activeSet.begin(); it1 != activeSet.end(); ++it1, ++i){
			cov.coeffRef(i,i) = calcDerivativeDiagElement(i, type);
			unsigned int j = i + 1;
			std::list<int>::const_iterator it2 = it1;
			++it2;
			for(; it2 != activeSet.end(); ++it2, ++j){
				cov.coeffRef(i,j) = kernelFuncDerivativeToParam(*it1, *it2, type);
				cov.coeffRef(j,i) = cov.coeff(i,j);
			}
		}
	}else{
		// derivative has only diag elements
		for(int i = 0; i < nrOfInducingPoints; ++i){
			cov.coeffRef(i,i) = calcDerivativeDiagElement(i, type); // derivative of m_hyperParams[2]^2
		}
	}
}

template<typename KernelType, unsigned int nrOfParams>
void KernelBase<KernelType, nrOfParams>::newRandHyperParams(){
	for(unsigned int i = 0; i < nrOfParams; ++i){
		if(m_kernelParams.m_params[i]->hasMoreThanOneDim()){
			for(unsigned int j = 0; j < ClassKnowledge::amountOfDims(); ++j){
				m_kernelParams.m_params[i]->getValues()[j] = (*m_randomGaussians[i])();
			}
		}else{
			m_kernelParams.m_params[i]->getValues()[0] = (*m_randomGaussians[i])();
		}
	}
}

template<typename KernelType, unsigned int nrOfParams>
void KernelBase<KernelType, nrOfParams>::setSeed(const int seed){
	m_seed = seed;
	for(unsigned int i = 0; i < nrOfParams; ++i){
		m_randomGaussians[i]->setSeed((seed + 1) * (i+1) * 53667);
	}
}

template<typename KernelType, unsigned int nrOfParams>
void KernelBase<KernelType, nrOfParams>::addToHyperParams(const KernelType& params, const double factor){
	for(unsigned int i = 0; i < m_kernelParams.paramsAmount; ++i){
		if(m_kernelParams.m_params[i]->hasMoreThanOneDim() && params.m_params[i]->hasMoreThanOneDim()){
			for(unsigned int j = 0; j < ClassKnowledge::amountOfDims(); ++j){
				m_kernelParams.m_params[i]->getValues()[j] += params.m_params[i]->getValues()[i] * factor;
			}
		}else{
			m_kernelParams.m_params[i]->getValues()[0] += params.m_params[i]->getValue() * factor;
		}
	}
}


