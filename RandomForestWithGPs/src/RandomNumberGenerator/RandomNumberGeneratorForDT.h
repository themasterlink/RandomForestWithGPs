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
#include "../Utility/Util.h"
#include "../Base/Observer.h"
#include <Eigen/Dense>

class RandomNumberGeneratorForDT : public Observer {
public:

	using base_generator_type = GeneratorType; // generator type
	using uniform_distribution_int = boost::random::uniform_int_distribution<int>; // generator type
	using uniform_distribution_real = boost::uniform_real<real>; // generator type
	using variante_generator = boost::variate_generator<base_generator_type, uniform_distribution_int>;

	RandomNumberGeneratorForDT(const int dim, const int minUsedData, const int maxUsedData,
			const int amountOfData, const int seed, const int amountOfDataUsedPerTree);

	virtual ~RandomNumberGeneratorForDT();

	int getRandDim();

	int getRandAmountOfUsedData();

	int getRandNextDataEle();

	void setRandFromRange(const int min, const int max);

	void setRandForDim(const int min, const int max);

	int getRandFromRange();

	void setMinAndMaxForSplitInDim(const unsigned int dim, const real min, const real max);

	real getRandSplitValueInDim(const unsigned int dim);

	void update(Subject* caller, unsigned int event) override;

	bool useDim(const int dim) const{ return m_useDim[dim]; }

	bool useWholeDataSet() const{ return m_stepSize < 1; };

	int getStepSize(){ return m_stepSize; };

	unsigned int getRandStepOverStorage();

private:
	int m_stepSize;  // > 1 means no step size used

	base_generator_type m_generator;

	uniform_distribution_int m_uniformDistDimension;
	uniform_distribution_int m_uniformDistUsedData;
	uniform_distribution_int m_uniformDistData;
	uniform_distribution_int m_uniformStepOverStorage;

	uniform_distribution_int m_uniformDistRange;

	std::vector<uniform_distribution_real> m_uniformSplitValues;

	variante_generator m_varGenDimension;
	variante_generator m_varGenUsedData;
	variante_generator m_varGenData;
	variante_generator m_varGenStepOverStorage;

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

inline unsigned int RandomNumberGeneratorForDT::getRandStepOverStorage(){
	return m_varGenStepOverStorage();
}


inline void RandomNumberGeneratorForDT::setRandFromRange(const int min, const int max){
	m_uniformDistRange.param(uniform_distribution_int::param_type(min, max));
}

inline void RandomNumberGeneratorForDT::setRandForDim(const int min, const int max){
	m_varGenDimension.distribution().param(uniform_distribution_int::param_type(min, max));
}

inline int RandomNumberGeneratorForDT::getRandFromRange(){
	return m_uniformDistRange(m_generator);
}

inline real RandomNumberGeneratorForDT::getRandSplitValueInDim(const unsigned int dim){
	if(m_uniformSplitValues.size() > dim && m_useDim[dim]){
//		m_mutex.lock();
		const real val =  m_uniformSplitValues[dim](m_generator);
//		m_mutex.unlock();
		return val;
	}else{
		printError("The rand split value generator has not been set yet!");
		return 0.;
	}
}

#endif /* RANDOMNUMBERGENERATORFORDT_RANDOMNUMBERGENERATORFORDT_H_ */
