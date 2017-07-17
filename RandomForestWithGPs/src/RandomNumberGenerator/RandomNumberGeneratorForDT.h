/*
 * RandomNumberGenerator.h
 *
 *  Created on: 03.06.2016
 *      Author: Max
 */

#ifndef RANDOMNUMBERGENERATORFORDT_RANDOMNUMBERGENERATORFORDT_H_
#define RANDOMNUMBERGENERATORFORDT_RANDOMNUMBERGENERATORFORDT_H_

#include "../Utility/Util.h"
#include "../Base/Observer.h"
#include "RandomUniformNr.h"

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

	RandomNumberGeneratorForDT(const int dim, const int minUsedData, const int maxUsedData,
			const int amountOfData, const int seed, const BaggingInformation& baggingInformation, const bool useRealOnlineUpdate);

	virtual ~RandomNumberGeneratorForDT();

	int getRandDim();

	int getRandAmountOfUsedData();

	int getRandNextDataEle();

	void setRandFromRange(const int min, const int max);

	void setRandForDim(const int max);

	int getRandFromRange();

	void setMinAndMaxForSplitInDim(const unsigned int dim, const Real min, const Real max);

	Real getRandSplitValueInDim(const unsigned int dim);

	void update(Subject* caller, unsigned int event) override;

	bool useDim(const int dim) const{ return m_useDim[dim]; }

	bool useWholeDataSet() const{ return m_baggingInformation.useWholeDataSet(); };

	const BaggingInformation& getBaggingInfo(){ return m_baggingInformation; };

	unsigned int getRandStepOverStorage();

	const bool isRandStepOverStorageUsed() const { return m_uniformStepOverStorage.isUsed(); };

	const bool useRealOnlineUpdate() const { return m_useRealOnlineUpdate; };

private:
	const BaggingInformation& m_baggingInformation;
	unsigned int m_currentStepSize;

	GeneratorType m_generator;

	RandomUniformUnsignedNr m_uniformDistDimension;
	RandomUniformNr m_uniformDistUsedData;
	RandomUniformUnsignedNr m_uniformDistData;
	RandomUniformUnsignedNr m_uniformStepOverStorage;
	RandomUniformNr m_uniformDistRange;

	std::vector<RandomDistributionReal> m_uniformSplitValues;

	Mutex m_mutex;

	std::vector<bool> m_useDim;

	const bool m_useRealOnlineUpdate;
};

inline int RandomNumberGeneratorForDT::getRandDim(){
	return m_uniformDistDimension();
}

inline int RandomNumberGeneratorForDT::getRandAmountOfUsedData(){
	return m_uniformDistUsedData();
}

inline int RandomNumberGeneratorForDT::getRandNextDataEle(){
	return m_uniformDistData();
}

inline unsigned int RandomNumberGeneratorForDT::getRandStepOverStorage(){
	return m_uniformStepOverStorage() + 1; // never returns zero
}


inline void RandomNumberGeneratorForDT::setRandFromRange(const int min, const int max){
	m_uniformDistRange.setMinAndMax(min, max);
}

inline void RandomNumberGeneratorForDT::setRandForDim(const int max){
	m_uniformDistDimension.setMax(max);
}

inline int RandomNumberGeneratorForDT::getRandFromRange(){
	return m_uniformDistRange();
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
