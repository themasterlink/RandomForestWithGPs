/*
 * DataReader.cc
 *
 *  Created on: 04.06.2016
 *      Author: Max
 */

#include "DataReader.h"

DataReader::DataReader(){
}

DataReader::~DataReader(){
}

void DataReader::readFromFile(Data& data, Labels& label, const std::string& inputName){
	std::string line;
	std::ifstream input(inputName);
	if(input.is_open()){
		while(std::getline(input, line)){
			std::vector<std::string> elements;
			std::stringstream ss(line);
			std::string item;
			while(std::getline(ss, item, ',')){
				elements.push_back(item);
			}
			DataElement newEle(elements.size() - 1);
			for(int i = 0; i < elements.size() - 1; ++i){
				newEle[i] = std::stod(elements[i]);
			}
			label.push_back(std::stoi(elements.back()) > 0 ? 1 : 0);
			data.push_back(newEle);
		}
		input.close();
	}else{
		printError("File was not found: " << inputName);
	}
}

