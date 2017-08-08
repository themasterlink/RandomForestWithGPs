/*
 * RandomNumberGeneratorForDT.cc
 *
 *  Created on: 03.06.2016
 *      Author: Max
 */

#include <regex>
#include "../RandomForests/OnlineRandomForest.h"

RandomNumberGeneratorForDT::RandomNumberGeneratorForDT(const int dim, const int minUsedData,
													   const int maxUsedData, const int amountOfData, const int seed,
													   const BaggingInformation& baggingInformation,
													   const bool useRealOnlineUpdate)
		: m_baggingInformation(baggingInformation),
		  m_currentStepSize(2),
		  m_generator((unsigned int) seed + 307),
		  m_uniformDistDimension(dim - 1, seed + 563), // 0 ... (dimension of data - 1)
		  m_uniformDistUsedData(minUsedData, maxUsedData, seed + 1093),
		  m_uniformDistData(amountOfData - 1, seed + 4933),
		  m_uniformStepOverStorage(amountOfData - 1, seed + 463),
		  m_uniformDistRange(0, 1, seed + 107),
		  m_useDim((unsigned long) dim, false),
		  m_useRealOnlineUpdate(useRealOnlineUpdate){
}

RandomNumberGeneratorForDT::~RandomNumberGeneratorForDT(){
}

void RandomNumberGeneratorForDT::setMinAndMaxForSplitInDim(const unsigned int dim, const Real min, const Real max){
	m_useDim[dim] = min < max;
	if(m_useDim[dim]){
		m_uniformSplitValues[dim].setMinAndMax(min, max);
	}
}

void RandomNumberGeneratorForDT::update(Subject* caller, unsigned int event){
	UNUSED(event);
	if(caller != nullptr && caller->classType() == ClassTypeSubject::ONLINERANDOMFOREST){
		OnlineRandomForest* forest = dynamic_cast<OnlineRandomForest*>(caller);
		OnlineStorage<LabeledVectorX*>& storage = forest->getStorageRef();
		const unsigned int dim = storage.dim();
		if(storage.isInPoolMode() && !m_useRealOnlineUpdate){
			printError("The pool mode can not be active when the real online update is out!");
		}
		if(!useWholeDataSet()){
			// if in pool mode -> use storage size
			const auto size = !storage.isInPoolMode() && m_useRealOnlineUpdate ? storage.getAmountOfNew() : storage.size();
			m_currentStepSize = m_baggingInformation.m_stepSizeOverData;
			if(m_baggingInformation.useStepSize()){
				m_currentStepSize = m_baggingInformation.m_stepSizeOverData;
			}else if(m_baggingInformation.useTotalAmountOfPoints()){
				m_currentStepSize = (size / m_baggingInformation.m_totalUseOfData) * 2;
			}else{
				printError("This type is unknown here!");
			}
			m_currentStepSize = std::max((unsigned int) 1, std::min(m_currentStepSize, size));
			m_uniformStepOverStorage.setMax(m_currentStepSize);
		}
		if(m_uniformSplitValues.size() != dim){
			m_uniformSplitValues.resize(dim);
			m_useDim.resize(dim);
		}
		m_mutex.lock();
		const std::vector<Vector2>& minMaxValues = forest->getMinMaxValues();
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

RandomNumberGeneratorForDT::BaggingInformation::BaggingInformation():
		m_stepSizeOverData(Settings::instance().getDirectValue < unsigned int>("OnlineRandomForest.Tree.Bagging.stepSizeOverData")),
m_totalUseOfData(Settings::instance().getDirectValue<unsigned int>("OnlineRandomForest.Tree.Bagging.totalAmountOfDataUsedPerTree")),
m_mode(BaggingInformation::getMode(Settings::instance().getDirectValue<std::string>("OnlineRandomForest.Tree.Bagging.mode"))){}

RandomNumberGeneratorForDT::BaggingInformation::BaggingMode
RandomNumberGeneratorForDT::BaggingInformation::getMode(const std::string& settingsField){
	try{
		std::regex stepSize("(\\s)*(use|Use)?(\\s)*(step|Step)(\\s)*(size|Size)(\\s)*"); // step size| stepsize
		std::regex totalUseOfData("(\\s)*(use|Use)?(\\s)*(total|Total)(\\s)*(use|Use|amount|Amount)(\\s)*((of|Of)(\\s)*(data|Data)(\\s)*)?");
		std::regex useWholeDataSet("(\\s)*(use|Use)?(\\s)*(all|All|whole|Whole)(\\s)*(data|Data)(\\s)*((set|Set|point|Point)(s)?)?(\\s)*");
		if(std::regex_match(settingsField, stepSize)){
			return BaggingMode::STEPSIZE;
		}else if(std::regex_match(settingsField,totalUseOfData)){
			return BaggingMode::TOTALUSEOFDATA;
		}else if(std::regex_match(settingsField, useWholeDataSet)){
			return BaggingMode::USEWHOLEDATASET;
		}else{
			printErrorAndQuit("This type is not supported here: " << settingsField);
			return BaggingMode::STEPSIZE;
		}
	}catch(std::exception& e){
		printErrorAndQuit("Regex was not correct: " << e.what());
	}
	return BaggingMode::STEPSIZE; // will never be executed
};

