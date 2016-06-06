/*
 * RandomForestWriter.h
 *
 *  Created on: 05.06.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_RANDOMFORESTWRITER_H_
#define RANDOMFORESTS_RANDOMFORESTWRITER_H_

#include "OtherRandomForest.h"
#include "../Utility/Util.h"

class RandomForestWriter{
public:
	static void writeToFile(const std::string& filePath, const OtherRandomForest& forest);

	static void readFromFile(const std::string& filePath, OtherRandomForest& forest);

private:
	RandomForestWriter();
	virtual ~RandomForestWriter();
};

#endif /* RANDOMFORESTS_RANDOMFORESTWRITER_H_ */
