/*
 * RandomUniformNr.h
 *
 *  Created on: 21.11.2016
 *      Author: Max
 */

#ifndef RANDOMNUMBERGENERATOR_RANDOMUNIFORMNR_H_
#define RANDOMNUMBERGENERATOR_RANDOMUNIFORMNR_H_

#include <stdint.h>
#include "../Utility/Util.h"

// from: https://en.wikipedia.org/wiki/Xorshift#Xorshift.2B

class RandomUniformNr {
public:

	RandomUniformNr(const int min, const int max, const int seed);
	~RandomUniformNr();

	void setSeed(const int seed);

	void setMinAndMax(const int min, const int max);

	// returns true if min and max are not equal
	bool isUsed() const{ return m_isUsed; };

	int operator()();

private:
	bool m_isUsed;

	uint64_t m_currentSeeds[2];
	int m_minAndDiff[2];

};


class RandomUniformUnsignedNr {
public:

	RandomUniformUnsignedNr(const unsigned int max, const int seed);
	~RandomUniformUnsignedNr() = default;

	void setSeed(const int seed);

	void setMax(const unsigned int max);

	unsigned int operator()();

	const bool isUsed() const { return m_diff > 0; }

private:

	uint64_t m_currentSeeds[2];
	unsigned int m_diff;
};

class RandomDistributionReal {
public:

	RandomDistributionReal(): dis(0._r,1._r){}

	RandomDistributionReal(const Real min, const Real max): dis(min,max){}

	void setMinAndMax(const Real min, const Real max);

	template<typename T>
	inline Real operator()(T& gen){ return dis(gen); };

private:

	std::uniform_real_distribution<Real> dis;

};

#endif /* RANDOMNUMBERGENERATOR_RANDOMUNIFORMNR_H_ */
