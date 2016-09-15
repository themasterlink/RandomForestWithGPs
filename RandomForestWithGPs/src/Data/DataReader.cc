/*
 * DataReader.cc
 *
 *  Created on: 04.06.2016
 *      Author: Max
 */

#include "DataReader.h"
#include <iostream>
#include "boost/filesystem.hpp"
#include "../Utility/ReadWriterHelper.h"


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

void DataReader::readFromFile(Data& data, const std::string& inputName){
	std::fstream input(inputName);
	if(input.is_open()){
		if(inputName.find(".binary") != std::string::npos){
			// is a binary file -> faster loading!
			ReadWriterHelper::readVector(input, data);
		}else{
			std::string line;
			while(std::getline(input, line)){
				std::vector<std::string> elements;
				std::stringstream ss(line);
				std::string item;
				while(std::getline(ss, item, ' ')){
					elements.push_back(item);
				}
				DataElement newEle(elements.size());
				for(int i = 0; i < elements.size(); ++i){
					newEle[i] = std::stod(elements[i]);
				}
				data.push_back(newEle);
			}
			input.close();
		}
	}else{
		printError("File was not found: " << inputName);
	}
}

void DataReader::readFromFiles(DataSets& dataSets, const std::string& folderLocation){
	boost::filesystem::path targetDir(folderLocation);
	boost::filesystem::directory_iterator end_itr;
	// cycle through the directory
	for(boost::filesystem::directory_iterator itr(targetDir); itr != end_itr; ++itr){
		if(boost::filesystem::is_directory(itr->path())){
			const std::string name(itr->path().filename().c_str());
			Data data;
			std::string filePath(itr->path().c_str());
			filePath += "/vectors.txt";
			readFromFile(data, filePath);
			dataSets.insert( std::pair<std::string, Data >(name, data));
		}
	}
}

