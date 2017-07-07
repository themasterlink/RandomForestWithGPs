/*
 * RandomUniformNr.cc
 *
 *  Created on: 21.11.2016
 *      Author: Max
 */

#include <cstdlib>
#include "RandomUniformNr.h"

RandomUniformNr::RandomUniformNr(const int min, const int max, const int seed){
	setSeed(seed);
	setMinAndMax(min, max);
}

RandomUniformNr::~RandomUniformNr() {
}


void RandomUniformNr::setSeed(const int seed){
	m_currentSeeds[0] = ((uint64_t) (abs(seed) + 1)) * 94940;
	m_currentSeeds[1] = ((uint64_t) (m_currentSeeds[0] + 19390)) * 877;
}

void RandomUniformNr::setMinAndMax(const int min, const int max){
	m_minAndDiff[0] = min;
	m_minAndDiff[1] = max - min;
	m_isUsed = min != max;
}

int RandomUniformNr::operator()(){
	uint64_t x = m_currentSeeds[0];
	const uint64_t y = m_currentSeeds[1];
	m_currentSeeds[0] = y;
	x ^= x << 23;
	m_currentSeeds[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
	return ((unsigned int) (m_currentSeeds[1] + y)) % m_minAndDiff[1] + m_minAndDiff[0];
}

RandomUniformUnsignedNr::RandomUniformUnsignedNr(const unsigned int max, const int seed): m_diff(max){
	setSeed(seed);
}

void RandomUniformUnsignedNr::setSeed(const int seed){
	m_currentSeeds[0] = ((uint64_t) (abs(seed) + 1)) * 94923;
	m_currentSeeds[1] = ((uint64_t) (m_currentSeeds[0] + 12996)) * 473;
}

unsigned int RandomUniformUnsignedNr::operator()(){
	uint64_t x = m_currentSeeds[0];
	const uint64_t y = m_currentSeeds[1];
	m_currentSeeds[0] = y;
	x ^= x << 23;
	m_currentSeeds[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
	return ((unsigned int) (m_currentSeeds[1] + y)) % m_diff;
}

void RandomUniformUnsignedNr::setMax(const unsigned int max){
	m_diff = max;
}

void RandomDistributionReal::setMinAndMax(const Real min, const Real max){
	dis.param(std::uniform_real_distribution<Real>::param_type(min, max));
}
