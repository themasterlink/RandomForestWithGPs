/*
 * RandomNumberGenerator.h
 *
 *  Created on: 03.06.2016
 *      Author: Max
 */

#ifndef RANDOMNUMBERGENERATORFORDT_RANDOMNUMBERGENERATORFORDT_H_
#define RANDOMNUMBERGENERATORFORDT_RANDOMNUMBERGENERATORFORDT_H_

#include <boost/random.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include "../Data/OnlineStorage.h"
#include "../Utility/Util.h"
#include "../Base/Observer.h"
#include <Eigen/Dense>

class RandomNumberGeneratorForDT : public Observer {
public:

	typedef boost::random::mt19937 base_generator_type; // generator type
	typedef boost::random::uniform_int_distribution<int> uniform_distribution_int; // generator type
	typedef boost::uniform_real<double> uniform_distribution_real; // generator type
	typedef boost::variate_generator<base_generator_type, uniform_distribution_int> variante_generator;

	RandomNumberGeneratorForDT(const int dim, const int minUsedData, const int maxUsedData,
			const int amountOfData, const int seed);

	virtual ~RandomNumberGeneratorForDT();

	int getRandDim();

	int getRandAmountOfUsedData();

	int getRandNextDataEle();

	void setRandFromRange(const int min, const int max);

	int getRandFromRange();

	double getRandSplitValueInDim(const unsigned int dim);

	void update(Subject* caller, unsigned int event);

	bool useDim(const int dim){ return m_useDim[dim]; }

private:
	base_generator_type m_generator;

	uniform_distribution_int m_uniformDistDimension;
	uniform_distribution_int m_uniformDistUsedData;
	uniform_distribution_int m_uniformDistData;

	uniform_distribution_int m_uniformDistRange;

	std::vector<uniform_distribution_real> m_uniformSplitValues;

	variante_generator m_varGenDimension;
	variante_generator m_varGenUsedData;
	variante_generator m_varGenData;

	boost::mutex m_mutex;

	std::vector<bool> m_useDim;
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
	m_uniformDistRange.param(uniform_distribution_int::param_type(min, max));
}

inline int RandomNumberGeneratorForDT::getRandFromRange(){
	return m_uniformDistRange(m_generator);
}

inline double RandomNumberGeneratorForDT::getRandSplitValueInDim(const unsigned int dim){
	if(m_uniformSplitValues.size() > dim){
		m_mutex.lock();
		const double val =  m_uniformSplitValues[dim](m_generator);
		m_mutex.unlock();
		return val;
	}else{
		printError("The rand split value generator has not been set yet!");
		return 0.;
	}
}

#endif /* RANDOMNUMBERGENERATORFORDT_RANDOMNUMBERGENERATORFORDT_H_ */
