/*
 * InformationPackage.h
 *
 *  Created on: 14.12.2016
 *      Author: Max
 */

#ifndef BASE_INFORMATIONPACKAGE_H_
#define BASE_INFORMATIONPACKAGE_H_

#include <boost/thread.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include "../Utility/StopWatch.h"

class ScreenOutput;
class ThreadMaster;

class InformationPackage {
	friend ScreenOutput;
	friend ThreadMaster;
public:
	enum InfoType { // low is more important, class can not be used ordering and values are important
		ORF_TRAIN = 0,
		ORF_TRAIN_FIX = 1, // can not be aborted
		IVM_TRAIN = 2,
		IVM_PREDICT = 3,
		IVM_RETRAIN = 4,
		IVM_MULTI_UPDATE = 5,
		IVM_INIT_DIFFERENCE_MATRIX = 6
	};

	InformationPackage(InfoType type, Real correctlyClassified, int amountOfPoints);

	Real correctlyClassified() const { return m_correctlyClassified; };

	int amountOfAffectedPoints() const { return m_amountOfAffectedPoints; };

	void changeCorrectlyClassified(const Real newValue){ m_correctlyClassified = newValue; };

	void changeAmountOfAffectedPoints(const int newAmount){ m_amountOfAffectedPoints = newAmount; };

	void performedOneTrainingStep(){ ++m_amountOfTrainingsSteps; };

	int amountOfTrainingStepsPerformed() const { return m_amountOfTrainingsSteps; };

	void finishedTask(){ m_performTask = true; };

	bool isTaskFinished(){ return m_performTask; };

	bool shouldTrainingBeAborted(){ return m_abortTraining; };

	void abortTraing(){ m_abortTraining = true; };

	void pauseTheTraining(){ m_shouldTrainingBeHold = true; };

	bool shouldTrainingBePaused(){ return m_shouldTrainingBeHold; }

	Real calcAttractionLevel(const int minAmountOfPoints, const int maxAmountOfPoints);

	bool isWaiting(){ return m_isWaiting; };

	void wait();

	void notify();

	InfoType getType(){ return m_type; };

	Real getWorkedAmountOfSeconds();

	void printLineToScreenForThisThread(const std::string& line);

	void overwriteLastLineToScreenForThisThread(const std::string& line);

	void setStandartInformation(const std::string& line){ m_lineMutex.lock(); m_standartInfo = line; m_lineMutex.unlock(); };

	std::string getStandartInformation(){ return m_standartInfo; };

	bool canBeAbortedAfterCertainTime();

	void setAdditionalInfo(const std::string& line);

	void setTrainingsTime(const Real maxTrainingsTime){ m_maxTrainingsTime = maxTrainingsTime; }

	Real getMaxTrainingsTime(){ return m_maxTrainingsTime;}

	int getPriority();

	Real runningTimeSinceLastWait(){ return m_sw.elapsedSeconds(); };

	bool canBeAbortedInGeneral();

private:

	const InfoType m_type;

	boost::interprocess::interprocess_condition m_condition;

	boost::interprocess::interprocess_mutex m_mutex;

	bool m_performTask;

	bool m_abortTraining;

	bool m_isWaiting;

	bool m_shouldTrainingBeHold;

	Real m_correctlyClassified;

	int m_amountOfAffectedPoints;

	int m_amountOfTrainingsSteps;

	Real m_workedTime;

	std::list<std::string> m_lines;

	std::string m_standartInfo;

	std::string m_additionalInformation;

	boost::mutex m_lineMutex;

	StopWatch m_sw;

	Real m_maxTrainingsTime;
};

#endif /* BASE_INFORMATIONPACKAGE_H_ */
