/*
 * BestHyperParams.h
 *
 *  Created on: 22.07.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_BESTHYPERPARAMS_H_
#define GAUSSIANPROCESS_BESTHYPERPARAMS_H_

#include <boost/thread.hpp> // Boost threads

class BestHyperParams {
public:
	BestHyperParams(const int amountOfMaxNoChange);
	~BestHyperParams();

	void trySet(int newRight, int newRightPositive, int newAmountOfValues, int newAmountOfCorLabels, double len, double sigmaF);

	void getBestHypParams(double& bestLen, double& bestSigmaF);

	void getBestParams(int& bestRight, int& bestRightPositive, int& bestAmountOfValues, int& bestAmountOfCorLabels, double& bestLen, double& bestSigmaF);

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
	double m_len;
	double m_sigmaF;
	int m_noChangeCounter;
	bool m_isFinish;
	bool m_shutDown;
	boost::mutex m_mutex;
};

#endif /* GAUSSIANPROCESS_BESTHYPERPARAMS_H_ */
