/*
 * RandomForestKernel.cc
 *
 *  Created on: 02.12.2016
 *      Author: Max
 */

#include "RandomForestKernel.h"

RandomForestKernel::RandomForestKernel(OnlineStorage<LabeledVectorX*>& storage, const int maxDepth, const int samplingAmount, const int amountOfUsedClasses, const bool createOrf):
	KernelBase<RandomForestKernelParams>(OwnKernelInitParams(maxDepth, samplingAmount, amountOfUsedClasses), false),
	m_heightSampler(3, maxDepth, 234389),
	m_rf(nullptr),
	m_mode(KernelMode::PARTITION){
	if(createOrf){
		m_rf = new OnlineRandomForest(storage, maxDepth, amountOfUsedClasses);
		OnlineRandomForest::TrainingsConfig trainingsConfig;
		trainingsConfig.m_mode = OnlineRandomForest::TrainingsConfig::TrainingsMode::TREEAMOUNT;
		trainingsConfig.m_amountOfTrees = (unsigned int) (samplingAmount + 1);
		m_rf->setTrainingsMode(trainingsConfig);
	}
}

RandomForestKernel::RandomForestKernel(OnlineStorage<LabeledVectorX*>& storage, const OwnKernelInitParams& initParams, const bool createOrf):
	KernelBase<RandomForestKernelParams>(initParams),
	m_heightSampler(3, initParams.m_maxDepth, 234389),
	m_rf(nullptr),
	m_mode(KernelMode::PARTITION){
	if(createOrf){
		m_rf = new OnlineRandomForest(storage, initParams.m_maxDepth, initParams.m_amountOfUsedClasses);
		OnlineRandomForest::TrainingsConfig trainingsConfig;
		trainingsConfig.m_mode = OnlineRandomForest::TrainingsConfig::TrainingsMode::TREEAMOUNT;
		trainingsConfig.m_amountOfTrees = (unsigned int) (initParams.m_samplingAmount + 1);
		m_rf->setTrainingsMode(trainingsConfig);
	}
}

RandomForestKernel::~RandomForestKernel(){
}

void RandomForestKernel::init(){
	m_init = true;
	if(Settings::getDirectBoolValue("RandomForestKernel.usePartitionInsteadOfLabels")){
		m_mode = KernelMode::PARTITION;
	}else{
		m_mode = KernelMode::LABEL;
	}
}

void RandomForestKernel::update(Subject* subject, unsigned int event){
	if(m_init){
		m_rf->update(subject, event);
		m_dataPoints = m_rf->getStorageRef().size();
	}
}

Real RandomForestKernel::calcDiagElement(unsigned int row) const{
	UNUSED(row);
	return 1; // because decision trees are deterministic
}

Real RandomForestKernel::calcDerivativeDiagElement(unsigned int row, const OwnKernelElement* type) const{
	UNUSED(row); UNUSED(type);
	return 0;
}

void RandomForestKernel::calcKernelVector(const VectorX& vector, const Matrix& dataMat, VectorX& res) const{
	res = VectorX(m_dataPoints);
	for(unsigned int i = 0; i < m_dataPoints; ++i){
		res.coeffRef(i) = (Real) kernelFuncVec(vector, dataMat.col(i));
	}
}

std::string RandomForestKernel::prettyString() const{
	std::stringstream str;
	str << m_kernelParams;
	return str.str();
}

Real RandomForestKernel::kernelFunc(const int row, const int col) const{
//	if(!m_calcedDifferenceMatrix){
		if(m_init){
			return kernelFuncVec(*m_rf->getStorageRef()[row], *m_rf->getStorageRef()[col]);
		}else{
			printError("The init process failed, init was tried: " << m_init);
		}
//	}else{
//		return (*m_differences)(row, col);
//	}
	return 0;
}

Real RandomForestKernel::kernelFuncVec(const VectorX& lhs, const VectorX& rhs) const{
	if(m_init){
		if(m_mode == KernelMode::LABEL){
			return m_rf->predict(lhs, rhs, (int) m_kernelParams.m_samplingAmount.getValue());
		}else if(m_mode == KernelMode::PARTITION){
			RandomUniformNr& sampler = const_cast<RandomUniformNr&>(m_heightSampler);
			return m_rf->predictPartitionEquality(lhs, rhs, sampler, (int) m_kernelParams.m_samplingAmount.getValue());
		}
	}
	return 0;
}

void RandomForestKernel::setSeed(const int seed){
	for(unsigned int i = 0; i < RandomForestKernelParams::paramsAmount; ++i){
		m_randomGaussians[i]->setSeed((seed + 1) * (i+1) * 53667);
	}
	m_heightSampler.setSeed((seed + 1) * 5233667);
}

Real RandomForestKernel::kernelFuncDerivativeToParam(const int row, const int col, const OwnKernelElement* type, const int element) const{
	UNUSED(row); UNUSED(col); UNUSED(type); UNUSED(element);
	return 0.;
}
