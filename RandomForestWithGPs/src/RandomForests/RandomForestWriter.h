/*
 * RandomForestWriter.h
 *
 *  Created on: 05.06.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_RANDOMFORESTWRITER_H_
#define RANDOMFORESTS_RANDOMFORESTWRITER_H_

#include "../Utility/Util.h"
#include "RandomForest.h"

class RandomForestWriter{
public:
	static void writeToFile(const std::string& filePath, const RandomForest& forest);

	static void readFromFile(const std::string& filePath, RandomForest& forest);

	static void writeToStream(std::fstream& file, const RandomForest& forest);

	static void readFromStream(std::fstream& file, RandomForest& forest);

private:
	RandomForestWriter();
	virtual ~RandomForestWriter();
};

#endif /* RANDOMFORESTS_RANDOMFORESTWRITER_H_ */
