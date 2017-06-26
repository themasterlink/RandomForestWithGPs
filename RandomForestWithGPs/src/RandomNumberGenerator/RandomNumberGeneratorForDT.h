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

class RandomNumberGeneratorForDT : public Observer {
public:

	class BaggingInformation {
	public:
		enum class BaggingMode {
			STEPSIZE,
			TOTALUSEOFDATA,
			USEWHOLEDATASET
		};

		BaggingInformation();

		const unsigned int m_stepSizeOverData;

		const unsigned int m_totalUseOfData;

		const bool useStepSize() const{ return m_mode == BaggingMode::STEPSIZE; };

		const bool useTotalAmountOfPoints() const{ return m_mode == BaggingMode::TOTALUSEOFDATA; };

		const bool useWholeDataSet() const { return m_mode == BaggingInformation::BaggingMode::USEWHOLEDATASET; };

	private:

		const BaggingMode m_mode;

		static BaggingMode getMode(const std::string& settingsField);
	};

	using base_generator_type = GeneratorType; // generator type
	using uniform_distribution_int = boost::random::uniform_int_distribution<int>; // generator type
	using uniform_distribution_real = boost::uniform_real<Real>; // generator type
	using variante_generator = boost::variate_generator<base_generator_type, uniform_distribution_int>;

	RandomNumberGeneratorForDT(const int dim, const int minUsedData, const int maxUsedData,
			const int amountOfData, const int seed, const BaggingInformation& baggingInformation, const bool useRealOnlineUpdate);

	virtual ~RandomNumberGeneratorForDT();

	int getRandDim();

	int getRandAmountOfUsedData();

	int getRandNextDataEle();

	void setRandFromRange(const int min, const int max);

	void setRandForDim(const int min, const int max);

	int getRandFromRange();

	void setMinAndMaxForSplitInDim(const unsigned int dim, const Real min, const Real max);

	Real getRandSplitValueInDim(const unsigned int dim);

	void update(Subject* caller, unsigned int event) override;

	bool useDim(const int dim) const{ return m_useDim[dim]; }

	bool useWholeDataSet() const{ return m_baggingInformation.useWholeDataSet(); };

	const BaggingInformation& getBaggingInfo(){ return m_baggingInformation; };

	unsigned int getRandStepOverStorage();

	const bool useRealOnlineUpdate() const { return m_useRealOnlineUpdate; };

private:
	const BaggingInformation& m_baggingInformation;
	unsigned int m_currentStepSize;

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

	Mutex m_mutex;

	std::vector<bool> m_useDim;

	const bool m_useRealOnlineUpdate;
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
	return (unsigned int) m_varGenStepOverStorage();
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

inline Real RandomNumberGeneratorForDT::getRandSplitValueInDim(const unsigned int dim){
	if(m_uniformSplitValues.size() > dim && m_useDim[dim]){
//		m_mutex.lock();
		const Real val =  m_uniformSplitValues[dim](m_generator);
//		m_mutex.unlock();
		return val;
	}else{
		printError("The rand split value generator has not been set yet!");
		return (Real) 0.;
	}
}

#endif /* RANDOMNUMBERGENERATORFORDT_RANDOMNUMBERGENERATORFORDT_H_ */
