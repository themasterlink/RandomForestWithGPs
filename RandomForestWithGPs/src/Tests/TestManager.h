//
// Created by denn_ma on 5/30/17.
//

#ifndef RANDOMFORESTWITHGPS_TESTMANAGER_H
#define RANDOMFORESTWITHGPS_TESTMANAGER_H

#include "TestInformation.h"
#include "../Data/LabeledVectorX.h"

class TestManager {

SingeltonMacro(TestManager);

public:

	void init();

	TestMode findMode(std::string& line);

	LabeledData getAllPointsFor(const std::string& defName, TestInformation::Instruction* scope);

	void removeClassesFrom(LabeledData& data, const TestInformation::TestDefineName& info);

	void run();

	int readAll();

	void setFilePath(const std::string& filePath){ m_filePath = filePath; };

	std::string getFilePath(){ return m_filePath; };

private:

	TestInformation m_testInformation;

	void performTest(const UniquePtr<OnlineRandomForest>& orf, const LabeledData& data);

	std::string m_filePath;
};


#endif //RANDOMFORESTWITHGPS_TESTMANAGER_H
