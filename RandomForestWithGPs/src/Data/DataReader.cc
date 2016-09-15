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

void DataReader::readFromFile(Data& data, Labels& label, const std::string& inputName, const int amountOfData){
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
			if(data.size() == amountOfData){
				break;
			}
		}
		input.close();
	}else{
		printError("File was not found: " << inputName);
	}
}

void DataReader::readFromFile(Data& data, const std::string& inputName, const int amountOfData){
	std::string inputPath(inputName);
	if(boost::filesystem::exists(inputName + ".binary")){
		// is a binary file -> faster loading!
		inputPath += ".binary";
		std::fstream input(inputPath, std::fstream::in);
		if(input.is_open()){
			long size;
			input.read((char*) &size, sizeof(long));
			data.resize(size);
			for(long i = 0; i < min(amountOfData,(int)size); ++i){
				ReadWriterHelper::readVector(input, data[i]);
			}
		}else{
			printError("The file could not be opened: " << inputPath);
		}
		input.close();
	}else if(boost::filesystem::exists(inputName + ".txt")){
		std::cout << "txt" << std::endl;
		inputPath += ".txt";
		std::ifstream input(inputPath);
		if(input.is_open()){
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
				if(data.size() == amountOfData){
					break;
				}
			}
			input.close();
		}else{
			printError("The file could not be opened: " << inputPath);
		}
	}else{
		printError("File was not found for .txt or .binary: " << inputName);
	}
}

void DataReader::readFromFiles(DataSets& dataSets, const std::string& folderLocation, const int amountOfData){
	boost::filesystem::path targetDir(folderLocation);
	boost::filesystem::directory_iterator end_itr;
	// cycle through the directory
	for(boost::filesystem::directory_iterator itr(targetDir); itr != end_itr; ++itr){
		if(boost::filesystem::is_directory(itr->path())){
			const std::string name(itr->path().filename().c_str());
			Data data;
			std::string filePath(itr->path().c_str());
			filePath += "/vectors";
			readFromFile(data, filePath, amountOfData);
			dataSets.insert( std::pair<std::string, Data >(name, data));
		}
	}
}

