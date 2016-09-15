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
	file.open(cpy, std::fstream::out | std::fstream::trunc);
	if(file.is_open()){
		long size = data.size();
		file.write((char*) &size, sizeof(long));
		std::cout << "Write size: " << size << std::endl;
		for(long i = 0; i < size; ++i){
			ReadWriterHelper::writeVector(file, data[i]);
		}
	}else{
		printError("This file could not be opened: " << cpy);
	}
	file.close();
}

