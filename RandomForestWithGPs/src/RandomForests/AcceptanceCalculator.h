//
// Created by denn_ma on 9/1/17.
//

#ifndef RANDOMFORESTWITHGPS_ACCEPTANCECALCULATOR_H
#define RANDOMFORESTWITHGPS_ACCEPTANCECALCULATOR_H

#include "../RandomNumberGenerator/RandomUniformNr.h"
#include "../RandomNumberGenerator/RandomGaussianNr.h"
#include "../RandomNumberGenerator/RandomExponentialNr.h"


class AcceptanceCalculator {

public:

	enum class AcceptanceMode{
		JUST_PERFORMANCE,
		GAUSSIAN,
		EXPONENTIAL_WHOLE,
		EXPONENTIAL_MIN_MAX,
		UNDEFINED
	};

	AcceptanceCalculator(const AcceptanceMode mode, const long seed);

	void setParams(const Real sd, const Real min, const Real max);

	Real calcAcceptance(const Real accuracy);

	static AcceptanceMode getModeForInput(const std::string& input);

	static bool stillUsePercent(AcceptanceMode mode);

private:

	const AcceptanceMode m_mode;

	RandomGaussianNr m_gaussianNr;

	RandomExponentialNr m_exponentialNr;

	Real m_sdOfForest;
	Real m_minOfForest;
	Real m_maxOfForest;

};


#endif //RANDOMFORESTWITHGPS_ACCEPTANCECALCULATOR_H
