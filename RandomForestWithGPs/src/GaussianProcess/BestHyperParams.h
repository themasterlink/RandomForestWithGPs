/*
 * BestHyperParams.h
 *
 *  Created on: 22.07.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_BESTHYPERPARAMS_H_
#define GAUSSIANPROCESS_BESTHYPERPARAMS_H_

#ifdef BUILD_OLD_CODE

#include <boost/thread.hpp> // Boost threads
#include "../Base/BaseType.h"
#include "../Base/Types.h"

class BestHyperParams {
public:
	BestHyperParams(const int amountOfMaxNoChange);
	~BestHyperParams();

	void trySet(int newRight, int newRightPositive, int newAmountOfValues, int newAmountOfCorLabels, Real len, Real sigmaF);

	void getBestHypParams(Real& bestLen, Real& bestSigmaF);

	void getBestParams(int& bestRight, int& bestRightPositive, int& bestAmountOfValues, int& bestAmountOfCorLabels, Real& bestLen, Real& bestSigmaF);

	void getNoChangeCounter(int& noChange);

	void getFinishLast(bool& isFinish);

	void getFinishDuring(bool& isFinish);

	bool checkGoal();

	const std::string prettyStringOfBest(const int precision = -1);

private:

	const int m_maxNrOfNoChange;
	int m_amountOfBestRight;
	int m_amountOfBestValues;
	int m_amountOfCorrectBestValues;
	int m_amountOfBestRightPositive;
	Real m_len;
	Real m_sigmaF;
	int m_noChangeCounter;
	bool m_isFinish;
	bool m_shutDown;
	Mutex m_mutex;
};

#endif // BUILD_OLD_CODE

#endif /* GAUSSIANPROCESS_BESTHYPERPARAMS_H_ */
