//
// Created by denn_ma on 5/30/17.
//

#ifndef RANDOMFORESTWITHGPS_TESTMANAGER_H
#define RANDOMFORESTWITHGPS_TESTMANAGER_H

#include "../Utility/Util.h"

class TestManager {

	enum class TestMode {
		LOAD = 0,
		DEFINE,
		TRAIN,
		TEST,
		UNDEFINED
	};

public:

	static void init(const std::string& filePath);

	static TestMode findMode(std::string& line);

	static void run();

private:
	TestManager() = delete;
	~TestManager() = delete;

//	static TestInformation m_testInformation;

};


#endif //RANDOMFORESTWITHGPS_TESTMANAGER_H
