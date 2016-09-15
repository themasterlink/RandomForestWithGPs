/*
 * DataBinaryWriter.cc
 *
 *  Created on: 15.09.2016
 *      Author: Max
 */

#include "DataBinaryWriter.h"
#include "../Utility/ReadWriterHelper.h"

DataBinaryWriter::DataBinaryWriter() {
	// TODO Auto-generated constructor stub

}

DataBinaryWriter::~DataBinaryWriter() {
	// TODO Auto-generated destructor stub
}

void DataBinaryWriter::toFile(const Data& data, const std::string& filePath){
	std::fstream file;
	std::string cpy(filePath);
	if(cpy.find_last_of(".")  != std::string::npos){
		cpy.erase(cpy.find_last_of("."));
		cpy += ".binary";
	}else{
		printError("The filetype should be set in: " << filePath);
	}
	file.open(cpy);
	if(file.is_open()){
		ReadWriterHelper::writeVector(file, data);
	}else{
		printError("This file could not be opened: " << cpy);
	}
	file.close();
}

