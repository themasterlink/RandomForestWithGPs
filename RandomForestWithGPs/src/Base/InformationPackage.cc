/*
 * InformationPackage.cc
 *
 *  Created on: 14.12.2016
 *      Author: Max
 */

#include "InformationPackage.h"
#include "../Utility/Util.h"
#include "Logger.h"
#include "Settings.h"
#include "CommandSettings.h"

InformationPackage::InformationPackage(InfoType type,
		Real correctlyClassified,
		int amountOfPoints): m_type(type),
		m_performTask(false),
		m_abortTraining(false),
		m_isWaiting(false),
		m_shouldTrainingBeHold(false),
		m_correctlyClassified(correctlyClassified),
		m_amountOfAffectedPoints(amountOfPoints),
		m_amountOfTrainingsSteps(0),
		m_workedTime(0),
		m_maxTrainingsTime(-1){
};


Real InformationPackage::calcAttractionLevel(const int minAmountOfPoints, const int maxAmountOfPoints){
	Real partAmount = 100.; // if all have the same size, just ignore this value and go after the correct classified amount
	if(maxAmountOfPoints != minAmountOfPoints){
		partAmount = (((Real) m_amountOfAffectedPoints - minAmountOfPoints)
				/ (Real)(maxAmountOfPoints - minAmountOfPoints)) * 100.;
	}
	return partAmount + (100 - m_correctlyClassified);
}

void InformationPackage::wait(){
	if(!m_abortTraining){ // only wait if the training is not aborted
		m_shouldTrainingBeHold = false; // could be called, because the training should be hold
		m_isWaiting = true;
		m_workedTime += m_sw.elapsedSeconds();
		boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(m_mutex);
		m_condition.wait(lock);
	}
};

void InformationPackage::notify(){
	m_condition.notify_all();
	m_sw.startTime();
	m_isWaiting = false;
};

Real InformationPackage::getWorkedAmountOfSeconds(){
	if(!m_isWaiting){
		return m_sw.elapsedSeconds() + m_workedTime;
	}
	return m_workedTime; // if it is waiting at the moment
}

void InformationPackage::setAdditionalInfo(const std::string& line){
	m_lineMutex.lock();
	m_additionalInformation = line;
	m_lineMutex.unlock();
}

void InformationPackage::printLineToScreenForThisThread(const std::string& line) {
	m_lineMutex.lock();
	m_lines.push_back(line);
	Logger::addSpecialLineToFile(line, m_standartInfo);
	m_lineMutex.unlock();
}

void InformationPackage::overwriteLastLineToScreenForThisThread(const std::string& line){
	if(m_lines.size() > 0){
		m_lineMutex.lock();
		m_lines.back() = line;
		//Logger::addSpecialLineToFile("overwrite: "+line, m_standartInfo);
		m_lineMutex.unlock();
	}
}

bool InformationPackage::canBeAbortedInGeneral(){
	switch(m_type){
	case ORF_TRAIN:
		return true;
	case ORF_TRAIN_FIX: // has a fix time, will end eventually
		return false;
	case IVM_TRAIN:
		return true;
	case IVM_PREDICT:
		return false;
	case IVM_RETRAIN:
		return true; // can not be aborted, will be aborted if all IVM_TRAIN are finished, or if the general abort signal is sended
	case IVM_MULTI_UPDATE:
		return true;
	case IVM_INIT_DIFFERENCE_MATRIX:
		return false;
	default:
		printError("This type is unknown!");
		return false;
	}

}

bool InformationPackage::canBeAbortedAfterCertainTime(){
	switch(m_type){
	case ORF_TRAIN:
		return true;
	case ORF_TRAIN_FIX:
		return false;
	case IVM_TRAIN:
		return true;
	case IVM_PREDICT:
		return false;
	case IVM_RETRAIN:
		return false; // can not be aborted, will be aborted if all IVM_TRAIN are finished
	case IVM_MULTI_UPDATE:
		return true;
	case IVM_INIT_DIFFERENCE_MATRIX:
		return false;
	default:
		printError("This type is unknown!");
		return false;
	}
}

int InformationPackage::getPriority(){
	switch(m_type){
	case ORF_TRAIN:
		return 6;
	case ORF_TRAIN_FIX:
		return 6;
	case IVM_MULTI_UPDATE: // should be higher than IVM_TRAIN
		return 5;
	case IVM_TRAIN:
		return 4;
	case IVM_PREDICT:
		return 2;
	case IVM_RETRAIN:
		return 1;
	case IVM_INIT_DIFFERENCE_MATRIX: // should be done before ivm start training
		return 7;
	default:
		printError("This type is unknown!");
		return 0;
	}
}

