/*
 * RFGPWriter.cc
 *
 *  Created on: 14.07.2016
 *      Author: Max
 */

#include "RFGPWriter.h"
#include "../Utility/Util.h"
#include "../Utility/ReadWriterHelper.h"
#include "../RandomForests/RandomForestWriter.h"
#include "../GaussianProcess/GaussianProcessWriter.h"

RFGPWriter::RFGPWriter() {
	// TODO Auto-generated constructor stub
}

RFGPWriter::~RFGPWriter() {
	// TODO Auto-generated destructor stub
}




void RFGPWriter::writeToFile(const std::string& filePath, RandomForestGaussianProcess& rfgp){
	if(filePath.length() == 0){
		printError("File path is empty!");
		return;
	}else if(rfgp.m_amountOfDataPoints == 0){
		printError("Number of data points of rfgp is zero -> writing not possible!");
		return;
	}
	std::fstream file(filePath,std::ios::out|std::ios::binary);
	if(file.is_open()){
		file.write((char*) &rfgp.m_amountOfDataPoints, sizeof(int));
		file.write((char*) &rfgp.m_amountOfUsedClasses, sizeof(int));
		for(int i = 0; i < rfgp.m_amountOfUsedClasses; ++i){
			size_t size = rfgp.m_classNames[i].size();
			file.write((char*) &size, sizeof(size_t));
			file.write(&rfgp.m_classNames[i][0],size);
		}
		for(int i = 0; i < rfgp.m_amountOfUsedClasses; ++i){
			std::cout << "Pure labels for " << rfgp.m_classNames[i] << ": " << rfgp.m_pureClassLabelForRfClass[i] << std::endl;
			file.write((char*) &rfgp.m_pureClassLabelForRfClass[i], sizeof(int));
		}
		// save forest:
		RandomForestWriter::writeToStream(file, rfgp.m_forest);
		for(int i = 0; i < rfgp.m_amountOfUsedClasses; ++i){
			for(int j = 0; j < rfgp.m_amountOfUsedClasses; ++j){
				int k = rfgp.m_gps[i][j] != NULL ? 1 : 0;
				file.write((char*) &(k), sizeof(int));
				std::cout << "K for " << rfgp.m_classNames[i] << " and " << rfgp.m_classNames[j] << " is: " << k << std::endl;
				if(rfgp.m_gps[i][j] != NULL){
					GaussianProcessWriter::writeToStream(file, *rfgp.m_gps[i][j]);
				}
			}
		}
		file.write((char*) &(rfgp.m_maxPointsUsedInGpSingleTraining), sizeof(int));
		file.close();
	}else{
		printError("The opening failed for: " << filePath);
		return;
	}
}

void RFGPWriter::readFromFile(const std::string& filePath, RandomForestGaussianProcess& rfgp){
	if(filePath.length() == 0){
		printError("File path is empty!");
		return;
	}
	std::fstream file(filePath,std::ios::binary| std::ios::in);
	if(file.is_open()){
		file.read((char*) &rfgp.m_amountOfDataPoints, sizeof(int));
		file.read((char*) &rfgp.m_amountOfUsedClasses, sizeof(int));
		for(int i = 0; i < rfgp.m_amountOfUsedClasses; ++i){
			size_t size = 0;
			file.read((char*)&size, sizeof(size_t));
			rfgp.m_classNames[i].resize(size);
			file.read(&rfgp.m_classNames[i][0], size);
			std::cout << "Class names: " << rfgp.m_classNames[i] << std::endl;
		}
		for(int i = 0; i < rfgp.m_amountOfUsedClasses; ++i){
			file.read((char*) &rfgp.m_pureClassLabelForRfClass[i], sizeof(int));
			std::cout << "Pure class for " << rfgp.m_classNames[i] << " is: " << rfgp.m_pureClassLabelForRfClass[i] << std::endl;
		}
		// save forest:
		printLine();
		RandomForestWriter::readFromStream(file, rfgp.m_forest);
		printLine();
		for(int i = 0; i < rfgp.m_amountOfUsedClasses; ++i){
			for(int j = 0; j < rfgp.m_amountOfUsedClasses; ++j){
				int k = 0;
				file.read((char*) &k, sizeof(int));
				std::cout << "K: " << k << std::endl;
				if(k == 1){
					std::cout << "Read: " << rfgp.m_classNames[i] << " with " << rfgp.m_classNames[j] << std::endl;
					rfgp.m_gps[i][j] = new GaussianProcess();
					GaussianProcessWriter::readFromStream(file, *rfgp.m_gps[i][j]);
				}
			}
		}
		file.read((char*) &(rfgp.m_maxPointsUsedInGpSingleTraining), sizeof(int));
		rfgp.m_didLoadTree = true;
		rfgp.m_nrOfRunningThreads = 0;
		file.close();
	}else{
		printError("File is not there: " << filePath);
	}
}
