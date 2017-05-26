/*
 * DataReader.h
 *
 *  Created on: 04.06.2016
 *      Author: Max
 */

#ifndef DATA_DATAREADER_H_
#define DATA_DATAREADER_H_

#include "LabeledVectorX.h"

class DataReader{

public:

	static void readFromFile(LabeledData& data, const std::string& inputName, const unsigned int amountOfData);

	static void readFromBinaryFile(LabeledData& data, const std::string& inputName, const unsigned int amountOfData);

	static void readFromFiles(DataSets& dataSets, const std::string& folderLocation, const unsigned int amountOfData, const bool readTxt, bool& didNormalizeData);

	static void readFromFile(LabeledData& data, const std::string& inputName, const unsigned int amountOfData, const unsigned int classNr, const bool readTxt = false, const bool containsDegrees = false);

private:

	DataReader();
	~DataReader();
};

#endif /* DATA_DATAREADER_H_ */
