/*
 * DataReader.h
 *
 *  Created on: 04.06.2016
 *      Author: Max
 */

#ifndef DATA_DATAREADER_H_
#define DATA_DATAREADER_H_

#include "ClassData.h"
#include "DataSets.h"

class DataReader{

public:

	static void readFromFile(ClassData& data, const std::string& inputName, const int amountOfData);

	static void readFromFiles(DataSets& dataSets, const std::string& folderLocation, const int amountOfData, const bool readTxt, bool& didNormalizeData);

private:

	static void readFromFile(ClassData& data, const std::string& inputName, const int amountOfData, const unsigned int classNr, const bool readTxt = false);

	DataReader();
	~DataReader();
};

#endif /* DATA_DATAREADER_H_ */
