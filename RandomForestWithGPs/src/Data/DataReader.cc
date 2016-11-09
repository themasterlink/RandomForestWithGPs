/*
 * DataReader.cc
 *
 *  Created on: 04.06.2016
 *      Author: Max
 */

#include "DataReader.h"
#include "DataBinaryWriter.h"
#include <iostream>
#include "../Utility/ReadWriterHelper.h"
#include "ClassKnowledge.h"

DataReader::DataReader(){
}

DataReader::~DataReader(){
}

void DataReader::readFromFile(ClassData& data, const std::string& inputName, const int amountOfData){
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
			ClassPoint* newEle = new ClassPoint(elements.size() - 1, std::stoi(elements.back()) > 0 ? 1 : 0);
			for(int i = 0; i < elements.size() - 1; ++i){
				(*newEle)[i] = std::stod(elements[i]);
			};
			data.push_back(newEle);
			if(data.size() == amountOfData){
				break;
			}
		}
		if(data.size() > 0 && ClassKnowledge::amountOfDims() == 0){
			ClassKnowledge::setAmountOfDims(data[0]->rows());
		}
		input.close();
	}else{
		printError("File was not found: " << inputName);
	}
}

void DataReader::readFromFile(ClassData& data, const std::string& inputName,
		const int amountOfData, const unsigned int classNr, const bool readTxt){
	std::string inputPath(inputName);
	if(boost::filesystem::exists(inputName + ".binary") && !readTxt){
		// is a binary file -> faster loading!
		inputPath += ".binary";
		std::fstream input(inputPath, std::fstream::in);
		if(input.is_open()){
			long size;
			input.read((char*) &size, sizeof(long));
			data.resize(min(amountOfData,(int)size));
			for(long i = 0; i < min(amountOfData,(int)size); ++i){
				data[i] = new ClassPoint();
				ReadWriterHelper::readPoint(input, *data[i]);
				data[i]->setLabel(classNr);
			}
		}else{
			printError("The file could not be opened: " << inputPath);
		}
		input.close();
	}else if(boost::filesystem::exists(inputName + ".txt")){
		std::cout << "Read txt" << std::endl;
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
				ClassPoint* newEle = new ClassPoint(elements.size(), classNr);
				for(int i = 0; i < elements.size(); ++i){
					(*newEle)[i] = std::stod(elements[i]);
				}
				data.push_back(newEle);
				if(data.size() == amountOfData){
					break;
				}
			}
			input.close();
			DataBinaryWriter::toFile(data, inputName + ".binary"); // create binary to avoid rereading .txt
		}else{
			printError("The file could not be opened: " << inputPath);
		}
	}else{
		printError("File was not found for .txt or .binary: " << inputName);
	}
	if(data.size() > 0 && ClassKnowledge::amountOfDims() == 0){
		ClassKnowledge::setAmountOfDims(data[0]->rows());
	}
}

void DataReader::readFromFiles(DataSets& dataSets, const std::string& folderLocation, const int amountOfData, const bool readTxt){
	boost::filesystem::path targetDir(folderLocation);
	boost::filesystem::directory_iterator end_itr;
	// cycle through the directory
	unsigned int label = 0;
	for(boost::filesystem::directory_iterator itr(targetDir); itr != end_itr; ++itr){
		if(boost::filesystem::is_directory(itr->path())){
			const std::string name(itr->path().filename().c_str());
			ClassData data;
			std::string filePath(itr->path().c_str());
			filePath += "/vectors";
			readFromFile(data, filePath, amountOfData, label, readTxt);
			++label;
			dataSets.insert( DataSetPair(name, data));
		}
	}
}

