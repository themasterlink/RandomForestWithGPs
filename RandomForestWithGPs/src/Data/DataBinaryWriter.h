/*
 * DataBinaryWriter.h
 *
 *  Created on: 15.09.2016
 *      Author: Max
 */

#ifndef DATA_DATABINARYWRITER_H_
#define DATA_DATABINARYWRITER_H_

#include "LabeledVectorX.h"

class DataBinaryWriter {
public:

	static void toFile(const Data& data, const std::string& filePath);

	static void toFile(const LabeledData& data, const std::string& filePath);

private:
	DataBinaryWriter();
	virtual ~DataBinaryWriter();
};

#endif /* DATA_DATABINARYWRITER_H_ */
