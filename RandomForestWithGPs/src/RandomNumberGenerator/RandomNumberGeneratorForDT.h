/*
 * RandomNumberGenerator.h
 *
 *  Created on: 03.06.2016
 *      Author: Max
 */

#ifndef RANDOMNUMBERGENERATORFORDT_RANDOMNUMBERGENERATORFORDT_H_
#define RANDOMNUMBERGENERATORFORDT_RANDOMNUMBERGENERATORFORDT_H_

#include "boost/random.hpp"
#include <boost/random/uniform_int.hpp>

typedef boost::random::mt19937 base_generator_type; // generator type
typedef boost::random::uniform_int_distribution<int> uniform_distribution_int; // generator type
typedef boost::variate_generator<base_generator_type, uniform_distribution_int> variante_generator;

class RandomNumberGeneratorForDT{
public:
	RandomNumberGeneratorForDT(const int dim, const int minUsedData, const int maxUsedData,
			const int amountOfData, const int seed);
	virtual ~RandomNumberGeneratorForDT();

	int getRandDim();

	int getRandAmountOfUsedData();

	int getRandNextDataEle();

	void setRandFromRange(const int min, const int max);

	int getRandFromRange();
private:
	base_generator_type m_generator;

	uniform_distribution_int m_uniformDistDimension;
	uniform_distribution_int m_uniformDistUsedData;
	uniform_distribution_int m_uniformDistData;

	uniform_distribution_int m_uniformDistRange;

	variante_generator m_varGenDimension;
	variante_generator m_varGenUsedData;
	variante_generator m_varGenData;

};

inline int RandomNumberGeneratorForDT::getRandDim(){
	return m_varGenDimension();
}

inline int RandomNumberGeneratorForDT::getRandAmountOfUsedData(){
	return m_varGenUsedData();
}

inline int RandomNumberGeneratorForDT::getRandNextDataEle(){
	return m_varGenData();
}

inline void RandomNumberGeneratorForDT::setRandFromRange(const int min, const int max){
	m_uniformDistRange.param(boost::random::uniform_int_distribution<int>::param_type(min, max));
}

inline int RandomNumberGeneratorForDT::getRandFromRange(){
	return m_uniformDistRange(m_generator);
}

#endif /* RANDOMNUMBERGENERATORFORDT_RANDOMNUMBERGENERATORFORDT_H_ */
