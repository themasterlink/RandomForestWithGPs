//
// Created by denn_ma on 5/30/17.
//

#ifndef RANDOMFORESTWITHGPS_TESTMANAGER_H
#define RANDOMFORESTWITHGPS_TESTMANAGER_H

#include "TestInformation.h"
#include "../Data/LabeledVectorX.h"

class TestManager {
public:

	static void init();

	static TestMode findMode(std::string& line);

	static LabeledData getAllPointsFor(const std::string& defName);

	static void removeClassesFrom(LabeledData& data, const TestInformation::TestDefineName& info);

	static void run();

	static int readAll();

	static void setFilePath(const std::string& filePath){ m_filePath = filePath; };

	static std::string getFilePath(){ return m_filePath; };

private:
	TestManager() = delete;
	~TestManager() = delete;

	static TestInformation m_testInformation;

	static void performTest(const std::unique_ptr<OnlineRandomForest>& orf, const LabeledData& data);

	static std::string m_filePath;
};


#endif //RANDOMFORESTWITHGPS_TESTMANAGER_H
