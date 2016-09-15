/*
 * DataReader.h
 *
 *  Created on: 04.06.2016
 *      Author: Max
 */

#ifndef DATA_DATAREADER_H_
#define DATA_DATAREADER_H_

#include "Data.h"

class DataReader{

public:

	static void readFromFile(Data& data, Labels& label, const std::string& inputName, const int amountOfData);

	static void readFromFiles(DataSets& dataSets, const std::string& folderLocation, const int amountOfData);

	static void readFromFile(Data& data, const std::string& inputName, const int amountOfData);

private:
	DataReader();
	~DataReader();
};

#endif /* DATA_DATAREADER_H_ */
