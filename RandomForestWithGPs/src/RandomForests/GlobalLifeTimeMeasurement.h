//
// Created by denn_ma on 8/31/17.
//

#ifndef RANDOMFORESTWITHGPS_GLOBALLIFETIMEMEASUREMENT_H
#define RANDOMFORESTWITHGPS_GLOBALLIFETIMEMEASUREMENT_H

#include <list>
#include "../Utility/Util.h"

class GlobalLifeTimeMeasurement : public Singleton<GlobalLifeTimeMeasurement> {

	friend class Singleton<GlobalLifeTimeMeasurement>;

public:

	unsigned int addNewTreeId();

	void dieTreeId(unsigned int treeId);

	void setPerformance(unsigned int treeId, const Real performance);

	void setRoundCounter(int roundCounter){ lockStatementWith(m_roundCounter = roundCounter, m_mutex); }

	bool isUsed(){ return m_roundCounter != -1; }

	int getRoundCounter() { return m_roundCounter; }

	void endAllTrees();

	void writeToFile(const std::string& fileName);

private:
	using LifeTimePerformance = std::pair<Real, unsigned int>;
	using LifeTimePair = std::pair<unsigned int, LifeTimePerformance>;
	using LifeTimeList = std::list<LifeTimePair>;

	LifeTimeList m_livingList;

	LifeTimeList m_deathList;

	Mutex m_mutex;

	int m_roundCounter{-1};

	unsigned int m_idCounter{0};

	GlobalLifeTimeMeasurement() = default;
};


#endif //RANDOMFORESTWITHGPS_GLOBALLIFETIMEMEASUREMENT_H
