/*
 * BestHyperParams.cc
 *
 *  Created on: 22.07.2016
 *      Author: Max
 */


#ifdef BUILD_OLD_CODE

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

void BestHyperParams::trySet(int newRight, int newRightPositive, int newAmountOfValues, int newAmountOfCorLabels, Real len, Real sigmaF){
	m_mutex.lock();
	if (newRight / (Real) newAmountOfValues
			+ newRightPositive / (Real) newAmountOfCorLabels
			> m_amountOfBestRight / (Real) m_amountOfBestValues
					+ m_amountOfBestRightPositive
							/ (Real) m_amountOfCorrectBestValues) {
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


void BestHyperParams::getBestParams(int& bestRight, int& bestRightPositive, int& bestAmountOfValues, int& bestAmountOfCorLabels, Real& bestLen, Real& bestSigmaF){
	m_mutex.lock();
	bestRight = m_amountOfBestRight;
	bestRightPositive = m_amountOfBestRightPositive;
	bestAmountOfValues = m_amountOfBestValues;
	bestAmountOfCorLabels = m_amountOfCorrectBestValues;
	bestLen = m_len;
	bestSigmaF = m_sigmaF;
	m_mutex.unlock();
}


void BestHyperParams::getBestHypParams(Real& bestLen, Real& bestSigmaF){
	lockStatementWith(bestLen = m_len;
							  bestSigmaF = m_sigmaF, m_mutex);
}

void BestHyperParams::getNoChangeCounter(int& noChange){
	lockStatementWith(noChange = m_noChangeCounter, m_mutex);
}

void BestHyperParams::getFinishLast(bool& isFinish){
	lockStatementWith(isFinish = m_isFinish || m_shutDown, m_mutex);
}

void BestHyperParams::getFinishDuring(bool& isFinish){
	lockStatementWith(isFinish = m_isFinish, m_mutex);
}

bool BestHyperParams::checkGoal(){
	m_mutex.lock();
	if(m_amountOfBestRight / (Real) m_amountOfBestValues * 100.0 > 95.0
			&& m_amountOfBestRightPositive / (Real) m_amountOfCorrectBestValues * 100.0 > 85.0){
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
		str << StringHelper::number2String(m_len, precision) << ", " << StringHelper::number2String(m_sigmaF, precision) << ", with: "
				<< StringHelper::number2String(m_amountOfBestRight / (Real) m_amountOfBestValues * 100.0, precision)
				<< " %, just right: "
				<< StringHelper::number2String(m_amountOfBestRightPositive / (Real) m_amountOfCorrectBestValues * 100.0, precision) << " %";
		m_mutex.unlock();
		return str.str();
	}else{
		std::stringstream str;
		m_mutex.lock();
		str << m_len << ", " << m_sigmaF << ", with: " << m_amountOfBestRight / (Real) m_amountOfBestValues * 100.0
				<< " %, just right: " << m_amountOfBestRightPositive / (Real) m_amountOfCorrectBestValues * 100.0 << " %";
		m_mutex.unlock();
		return str.str();
	}
}

#endif // BUILD_OLD_CODE