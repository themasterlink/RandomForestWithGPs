/*
 * BestHyperParams.cc
 *
 *  Created on: 22.07.2016
 *      Author: Max
 */

#include "BestHyperParams.h"
#include "../Utility/Util.h"

BestHyperParams::BestHyperParams(const int amountOfMaxNoChange):
	m_maxNrOfNoChange(amountOfMaxNoChange),
	m_amountOfBestRight(0),
	m_amountOfBestValues(1),
	m_amountOfCorrectBestValues(1),
	m_amountOfBestRightPositive(0),
	m_len(70),
	m_sigmaF(0.5),
	m_noChangeCounter(1),
	m_isFinish(false),
	m_shutDown(false){
}

BestHyperParams::~BestHyperParams(){
}

void BestHyperParams::trySet(int newRight, int newRightPositive, int newAmountOfValues, int newAmountOfCorLabels, double len, double sigmaF){
	m_mutex.lock();
	if (newRight / (double) newAmountOfValues
			+ newRightPositive / (double) newAmountOfCorLabels
			> m_amountOfBestRight / (double) m_amountOfBestValues
					+ m_amountOfBestRightPositive
							/ (double) m_amountOfCorrectBestValues) {
		m_amountOfBestRight = newRight;
		m_amountOfBestRightPositive = newRightPositive;
		m_amountOfBestValues = newAmountOfValues;
		m_amountOfCorrectBestValues = newAmountOfCorLabels;
		m_len = len;
		m_sigmaF = sigmaF;
		m_noChangeCounter = 0;
	}
	m_noChangeCounter += 1;
	if(m_noChangeCounter > m_maxNrOfNoChange){
		m_shutDown = true;
	}
	m_mutex.unlock();
}


void BestHyperParams::getBestParams(int& bestRight, int& bestRightPositive, int& bestAmountOfValues, int& bestAmountOfCorLabels, double& bestLen, double& bestSigmaF){
	m_mutex.lock();
	bestRight = m_amountOfBestRight;
	bestRightPositive = m_amountOfBestRightPositive;
	bestAmountOfValues = m_amountOfBestValues;
	bestAmountOfCorLabels = m_amountOfCorrectBestValues;
	bestLen = m_len;
	bestSigmaF = m_sigmaF;
	m_mutex.unlock();
}


void BestHyperParams::getBestHypParams(double& bestLen, double& bestSigmaF){
	m_mutex.lock();
	bestLen = m_len;
	bestSigmaF = m_sigmaF;
	m_mutex.unlock();
}

void BestHyperParams::getNoChangeCounter(int& noChange){
	m_mutex.lock();
	noChange = m_noChangeCounter;
	m_mutex.unlock();
}

void BestHyperParams::getFinishLast(bool& isFinish){
	m_mutex.lock();
	isFinish = m_isFinish || m_shutDown;
	m_mutex.unlock();
}

void BestHyperParams::getFinishDuring(bool& isFinish){
	m_mutex.lock();
	isFinish = m_isFinish;
	m_mutex.unlock();
}

bool BestHyperParams::checkGoal(){
	m_mutex.lock();
	if(m_amountOfBestRight / (double) m_amountOfBestValues * 100.0 > 95.0
			&& m_amountOfBestRightPositive / (double) m_amountOfCorrectBestValues * 100.0 > 85.0){
		m_isFinish = true;
		m_mutex.unlock();
		return true;
	}
	m_mutex.unlock();
	return false;
}

const std::string BestHyperParams::prettyStringOfBest(const int precision){
	if(precision != -1){
		std::stringstream str;
		m_mutex.lock();
		str << number2String(m_len, precision) << ", " << number2String(m_sigmaF, precision) << ", with: "
				<< number2String(m_amountOfBestRight / (double) m_amountOfBestValues * 100.0, precision)
				<< " %, just right: "
				<< number2String(m_amountOfBestRightPositive / (double) m_amountOfCorrectBestValues * 100.0, precision) << " %";
		m_mutex.unlock();
		return str.str();
	}else{
		std::stringstream str;
		m_mutex.lock();
		str << m_len << ", " << m_sigmaF << ", with: " << m_amountOfBestRight / (double) m_amountOfBestValues * 100.0
				<< " %, just right: " << m_amountOfBestRightPositive / (double) m_amountOfCorrectBestValues * 100.0 << " %";
		m_mutex.unlock();
		return str.str();
	}
}
