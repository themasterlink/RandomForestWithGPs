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

	void trySet(int newWright, double len, double sigmaF);

	void getBestParams(int& bestWright, double& bestLen, double& bestSigmaF);

	void getNoChangeCounter(int& noChange);

	void getFinishLast(bool& isFinish);

	void getFinishDuring(bool& isFinish);

	void reachedGoal();

private:
	const int m_maxNrOfNoChange;
	int m_amountOfBestWright;
	double m_len;
	double m_sigmaF;
	int m_noChangeCounter;
	bool m_isFinish;
	bool m_shutDown;
	boost::mutex m_mutex;
};

#endif /* GAUSSIANPROCESS_BESTHYPERPARAMS_H_ */
