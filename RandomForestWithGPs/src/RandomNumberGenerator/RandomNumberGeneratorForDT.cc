/*
 * RandomNumberGeneratorForDT.cc
 *
 *  Created on: 03.06.2016
 *      Author: Max
 */

#include "RandomNumberGeneratorForDT.h"
#include "../RandomForests/OnlineRandomForest.h"

RandomNumberGeneratorForDT::RandomNumberGeneratorForDT(const int dim, const int minUsedData,
		const int maxUsedData, const int amountOfData, const int seed)
		:	m_generator(seed),
			m_uniformDistDimension(0, dim - 1), // 0 ... (dimension of data - 1)
			m_uniformDistUsedData(minUsedData, maxUsedData),
			m_uniformDistData(0, amountOfData - 1),
			m_varGenDimension(m_generator, m_uniformDistDimension),
			m_varGenUsedData(m_generator, m_uniformDistUsedData),
			m_varGenData(m_generator, m_uniformDistData),
			m_useDim(dim, false){
}

RandomNumberGeneratorForDT::~RandomNumberGeneratorForDT(){
}

void RandomNumberGeneratorForDT::update(Subject* caller, unsigned int event){
	if(caller != nullptr && caller->classType() == ClassTypeSubject::ONLINERANDOMFOREST){
		OnlineRandomForest* forest = dynamic_cast<OnlineRandomForest*>(caller);
		OnlineStorage<ClassPoint*>& storage = forest->getStorageRef();
		const unsigned int dim = storage.dim();
		if(m_uniformSplitValues.size() != dim){
			m_uniformSplitValues.resize(dim);
			m_useDim.resize(dim);
		}
		m_mutex.lock();
		const std::vector<Eigen::Vector2d >& minMaxValues = forest->getMinMaxValues();
		for(unsigned int i = 0; i < dim; ++i){
			m_useDim[i] = minMaxValues[i][0] < minMaxValues[i][1];
			if(m_useDim[i]){
				m_uniformSplitValues[i].param(uniform_distribution_real::param_type(minMaxValues[i][0], minMaxValues[i][1]));
			}
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
