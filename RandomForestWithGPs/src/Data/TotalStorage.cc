/*
 * TotalStorage.cc
 *
 *  Created on: 14.10.2016
 *      Author: Max
 */

#include "TotalStorage.h"
#include "ClassKnowledge.h"
#include "DataReader.h"

TotalStorage::InternalStorage TotalStorage::m_storage;
ClassPoint TotalStorage::m_defaultEle;
unsigned int TotalStorage::m_totalSize(0);

TotalStorage::TotalStorage(){}

TotalStorage::~TotalStorage(){}

ClassPoint* TotalStorage::getData(unsigned int classNr, unsigned int elementNr){
	if(m_storage.size() > 0){
		Iterator it = m_storage.find(ClassKnowledge::getNameFor(classNr));
		if(it != m_storage.end()){
			return it->second[elementNr];
		}
	}
	return &m_defaultEle;
}

void TotalStorage::readData(const std::string& folderLocation, const int amountOfData){
	const bool readTxt = false;
	DataReader::readFromFiles(m_storage, folderLocation, amountOfData, readTxt);
}

ClassPoint* TotalStorage::getDefaultEle(){
	return &m_defaultEle;
}

unsigned int TotalStorage::getTotalSize(){
	return m_totalSize;
}

unsigned int TotalStorage::getSize(unsigned int classNr){
	Iterator it = m_storage.find(ClassKnowledge::getNameFor(classNr));
	if(it != m_storage.end()){
		it->second.size();
	}
	return 0;
}
