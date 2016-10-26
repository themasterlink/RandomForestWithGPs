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
			m_varGenData(m_generator, m_uniformDistData){
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
			m_minMaxValues.resize(dim);
			for(unsigned int i = 0; i < dim; ++i){
				m_minMaxValues[i][0] = DBL_MAX;
				m_minMaxValues[i][1] = -DBL_MAX;
			}
		}
		switch(event){
			case OnlineStorage<ClassPoint*>::APPEND:{
				ClassPoint& point = *storage.last();
				for(unsigned int k = 0; k < dim; ++k){
					bool change = false;
					if(point[k] < m_minMaxValues[k][0]){
						m_minMaxValues[k][0] = point[k];
						change = true;
					}
					if(point[k] > m_minMaxValues[k][1]){
						m_minMaxValues[k][1] = point[k];
						change = true;
					}
					if(change){
						m_uniformSplitValues[k].param(uniform_distribution_real::param_type(m_minMaxValues[k][0], m_minMaxValues[k][1]));
					}
				}
				break;
			}
			case OnlineStorage<ClassPoint*>::APPENDBLOCK:{
				const unsigned int start = storage.getLastUpdateIndex();
				const unsigned int dim = storage.dim();
				for(unsigned int t = start; t < storage.size(); ++t){
					ClassPoint& point = *storage[t];
					for(unsigned int k = 0; k < dim; ++k){
						if(point[k] < m_minMaxValues[k][0]){
							m_minMaxValues[k][0] = point[k];
						}
						if(point[k] > m_minMaxValues[k][1]){
							m_minMaxValues[k][1] = point[k];
						}
					}
				}
				for(unsigned int i = 0; i < dim; ++i){
					m_uniformSplitValues[i].param(uniform_distribution_real::param_type(m_minMaxValues[i][0], m_minMaxValues[i][1]));
				}
				break;
			}
			default:{
				printError("This event is not handled here!");
				break;
			}
		}
	}else{
		printError("This subject type is unknown!");
	}
}
