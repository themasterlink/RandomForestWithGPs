/*
 * BestHyperParams.cc
 *
 *  Created on: 22.07.2016
 *      Author: Max
 */

#include "BestHyperParams.h"

BestHyperParams::BestHyperParams(const int amountOfMaxNoChange):
	m_maxNrOfNoChange(amountOfMaxNoChange),
	m_amountOfBestWright(0),
	m_len(70),
	m_sigmaF(0.5),
	m_noChangeCounter(1),
	m_isFinish(false),
	m_shutDown(false){
}

BestHyperParams::~BestHyperParams(){
}

void BestHyperParams::trySet(int newWright, double len, double sigmaF){
	m_mutex.lock();
	if(newWright > m_amountOfBestWright){
		m_amountOfBestWright = newWright;
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


void BestHyperParams::getBestParams(int& bestWright, double& bestLen, double& bestSigmaF){
	m_mutex.lock();
	bestWright = m_amountOfBestWright;
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

void BestHyperParams::reachedGoal(){
	m_mutex.lock();
	m_isFinish = true;
	m_mutex.unlock();
}
