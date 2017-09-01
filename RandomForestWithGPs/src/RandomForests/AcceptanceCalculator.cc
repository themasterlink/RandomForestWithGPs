//
// Created by denn_ma on 9/1/17.
//

#include <regex>
#include "AcceptanceCalculator.h"

AcceptanceCalculator::AcceptanceCalculator(const AcceptanceMode mode, const long seed):
		m_mode(mode), m_gaussianNr(0, 1, seed), m_realUniform(0, 1), m_gen((seed+1) * 3){
	if(m_mode == AcceptanceMode::UNDEFINED){
		printError("This type is not supported!");
	}
	m_minOfForest = m_maxOfForest = -1;
}

void AcceptanceCalculator::setParams(const Real sd, const Real min, const Real max){
	m_sdOfForest = sd;
	m_minOfForest = min;
	m_maxOfForest = max;
	if(absReal(m_minOfForest - m_maxOfForest) < 1e-5_r){ // difference to small
		m_maxOfForest = m_minOfForest + 0.05_r; // increase
	}
	switch(m_mode){
		case AcceptanceMode::JUST_PERFORMANCE:{ // no random numbers needed
			break;
		}
		case AcceptanceMode::GAUSSIAN:{
			m_gaussianNr.reset(0, m_sdOfForest);
			break;
		}
		case AcceptanceMode::EXPONENTIAL_MIN_MAX: case AcceptanceMode::EXPONENTIAL_WHOLE: {
			m_realUniform.setMinAndMax(0, 1);
			break;
		}
		case AcceptanceMode::UNDEFINED:{
			printError("This type is undefined"); break;
		}
	}
}

Real AcceptanceCalculator::calcAcceptance(const Real accuracy){
	switch(m_mode){
		case AcceptanceMode::JUST_PERFORMANCE:{ // no random numbers needed
			return accuracy * 100.0_r; // convert in %
		}
		case AcceptanceMode::GAUSSIAN:{
			return std::min(100.0_r, accuracy * 100.0_r + absReal(m_gaussianNr())); // convert in %
		}
		case AcceptanceMode::EXPONENTIAL_MIN_MAX:{
			if(m_minOfForest >= 0 && m_maxOfForest >= 0){
				const auto minMaxAccuracy = (accuracy - m_minOfForest) / (m_maxOfForest - m_minOfForest);
				// min and max are required -> accuracy could be worse or better than any existing tree
				const auto lambda = 1.0_r - std::min(std::max(0.0_r, minMaxAccuracy), 1.0_r);
				return lambda * expReal(-lambda * m_realUniform(m_gen));
			}else{
				printError("The min and max values are not set!");
			}
		}
		case AcceptanceMode::EXPONENTIAL_WHOLE:{
			const auto lambda = 1.0_r - accuracy;
			return lambda * expReal(-lambda * m_realUniform(m_gen));
		}
		case AcceptanceMode::UNDEFINED:{
			printError("This type is undefined");
			break;
		}
	}
	return accuracy;
}

AcceptanceCalculator::AcceptanceMode AcceptanceCalculator::getModeForInput(const std::string& input){
	try{
		std::regex gaussian("(\\s)*(use|Use)?(\\s)*(gaussian|Gaussian|gauss|Gauss)(\\s)*((abs|absolute|Abs|Absolute)?(\\s)*(noise|Noise|Error|error))?(\\s)*");
		const std::string minMax("(min|Min)(\\s)*(and|And)?(\\s)*(max|Max)");
		std::regex exponentialMinMax("(\\s)*(use|Use)?(\\s)*(exponential|Exponential|exp|Exp)(\\s)*(with|With)(\\s)*" + minMax + "(\\s)*");
		std::regex exponential("(\\s)*(use|Use)?(\\s)*(exponential|Exponential|exp|Exp)(\\s)*((without|Without)(\\s)*" + minMax + "|((with|With)?(\\s)*(all|All|Whole|whole)))?(\\s)*");
		std::regex justPerformance("(\\s)*(Just|just|only|Only)?(\\s)*(use|Use)?(\\s)*(accuracy|Accuracy|performance|Performance)(\\s)*");
		if(std::regex_match(input, exponentialMinMax)){ // should be checked before exponential whole!
			return AcceptanceMode::EXPONENTIAL_MIN_MAX;
		}else if(std::regex_match(input, exponential)){
			return AcceptanceMode::EXPONENTIAL_WHOLE;
		}else if(std::regex_match(input, gaussian)){
			return AcceptanceMode::GAUSSIAN;
		}else if(std::regex_match(input, justPerformance)){
			return AcceptanceMode::JUST_PERFORMANCE;
		}else{
			printErrorAndQuit("This type is not supported here: " << input);
			return AcceptanceMode::UNDEFINED;
		}
	}catch(std::exception& e){
		printErrorAndQuit("Regex was not correct: " << e.what());
	}
	return AcceptanceMode::UNDEFINED;
}

bool AcceptanceCalculator::stillUsePercent(AcceptanceCalculator::AcceptanceMode mode){
	switch(mode){
		case AcceptanceMode::GAUSSIAN: case AcceptanceMode::JUST_PERFORMANCE: {
			return true;
		}
		case AcceptanceMode::EXPONENTIAL_MIN_MAX:
		case AcceptanceMode::EXPONENTIAL_WHOLE:{
			return false;
		}
		case AcceptanceMode::UNDEFINED:{
			printError("This type is undefined");
			return false;
		}
	}
}
