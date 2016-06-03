/*
 * RandomNumberGeneratorForDT.cc
 *
 *  Created on: 03.06.2016
 *      Author: Max
 */

#include "RandomNumberGeneratorForDT.h"


RandomNumberGeneratorForDT::RandomNumberGeneratorForDT(const int dim, const int minUsedData, const int maxUsedData, const int amountOfData, const int seed):
	m_generator(seed),
	m_uniformDistDimension(0,dim - 1), // 0 ... (dimension of data - 1)
		m_uniformDistUsedData(minUsedData, maxUsedData),
		m_uniformDistData(0, amountOfData - 1),
		m_varGenDimension(m_generator, m_uniformDistDimension),
		m_varGenUsedData(m_generator, m_uniformDistUsedData),
		m_varGenData(m_generator, m_uniformDistData){
}


RandomNumberGeneratorForDT::~RandomNumberGeneratorForDT(){
}
