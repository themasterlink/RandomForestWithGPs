/*
 * RandomNumberGeneratorForDT.cc
 *
 *  Created on: 03.06.2016
 *      Author: Max
 */

#include "RandomNumberGeneratorForDT.h"
#include "../RandomForests/OnlineRandomForest.h"

RandomNumberGeneratorForDT::RandomNumberGeneratorForDT(const int dim, const int minUsedData,
	const int maxUsedData, const int amountOfData, const int seed, const int amountOfDataUsedPerTree)
	:	m_stepSize(std::max(1,std::min(amountOfDataUsedPerTree, amountOfData))),
		m_generator(seed),
		m_uniformDistDimension(0, dim - 1), // 0 ... (dimension of data - 1)
		m_uniformDistUsedData(minUsedData, maxUsedData),
		m_uniformDistData(0, amountOfData - 1),
		m_uniformStepOverStorage(1, m_stepSize),
		m_varGenDimension(m_generator, m_uniformDistDimension),
		m_varGenUsedData(m_generator, m_uniformDistUsedData),
		m_varGenData(m_generator, m_uniformDistData),
		m_varGenStepOverStorage(m_generator, m_uniformStepOverStorage),
		m_useDim(dim, false){
}

RandomNumberGeneratorForDT::~RandomNumberGeneratorForDT(){
}

void RandomNumberGeneratorForDT::setMinAndMaxForSplitInDim(const unsigned int dim, const double min, const double max){
	m_useDim[dim] = min < max;
	if(m_useDim[dim]){
		m_uniformSplitValues[dim].param(uniform_distribution_real::param_type(min, max));
	}
}

void RandomNumberGeneratorForDT::update(Subject* caller, unsigned int event){
	UNUSED(event);
	if(caller != nullptr && caller->classType() == ClassTypeSubject::ONLINERANDOMFOREST){
		OnlineRandomForest* forest = dynamic_cast<OnlineRandomForest*>(caller);
		OnlineStorage<ClassPoint*>& storage = forest->getStorageRef();
		const unsigned int dim = storage.dim();
		if(!useWholeDataSet()){
			m_stepSize = (std::max(1,std::min(m_stepSize, (int) storage.size())));
			m_varGenStepOverStorage.distribution().param(uniform_distribution_int::param_type(1, m_stepSize));
		}
		if(m_uniformSplitValues.size() != dim){
			m_uniformSplitValues.resize(dim);
			m_useDim.resize(dim);
		}
		m_mutex.lock();
		const std::vector<Eigen::Vector2d >& minMaxValues = forest->getMinMaxValues();
		for(unsigned int i = 0; i < dim; ++i){
			setMinAndMaxForSplitInDim(i, minMaxValues[i][0], minMaxValues[i][1]);
//			else{
//				m_useDim[i] = false;
				// just to get any value -> else this will throw an execption
//				m_uniformSplitValues[i].param(uniform_distribution_real::param_type(minMaxValues[i][0], minMaxValues[i][0] + (minMaxValues[i][0] * 1e-5) + 1e-7));
//			}
		}
		m_mutex.unlock();
	}else{
		printError("This subject type is unknown!");
	}
}
