/*
 * TotalStorage.cc
 *
 *  Created on: 14.10.2016
 *      Author: Max
 */

#include "TotalStorage.h"
#include "ClassKnowledge.h"
#include "DataReader.h"
#include "../Base/Settings.h"
#include "../Base/ScreenOutput.h"
#include "../Base/CommandSettings.h"
#include "../Data/DataConverter.h"

TotalStorage::InternalStorage TotalStorage::m_storage;
ClassPoint TotalStorage::m_defaultEle;
unsigned int TotalStorage::m_totalSize(0);
DataPoint TotalStorage::m_center;
DataPoint TotalStorage::m_var;

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

void TotalStorage::readData(const int amountOfData){
	std::string folderLocation;
	if(CommandSettings::get_useFakeData()){
		Settings::getValue("TotalStorage.folderLocFake", folderLocation);
	}else{
		Settings::getValue("TotalStorage.folderLocReal", folderLocation);
	}
	const bool readTxt = false;
	bool didNormalizeStep = false;
	if(Settings::getDirectBoolValue("TotalStorage.readFromFolder")){
		DataReader::readFromFiles(m_storage, folderLocation, amountOfData, readTxt, didNormalizeStep);
	}else{
		ClassData data;
		DataReader::readFromBinaryFile(data, "../binary/dataFor_0.binary", amountOfData);
		for(unsigned int i = 0; i < data.size(); ++i){
			DataSetsIterator it = m_storage.find(ClassKnowledge::getNameFor(data[i]->getLabel()));
			if(it != m_storage.end()){
				it->second.push_back(data[i]);
			}else{
				ClassData newData;
				m_storage.insert(DataSetPair(ClassKnowledge::getNameFor(data[i]->getLabel()), newData));
				DataSetsIterator newIt = m_storage.find(ClassKnowledge::getNameFor(data[i]->getLabel()));
				if(newIt != m_storage.end()){
					newIt->second.push_back(data[i]);
				}
			}
		}
	}
	std::string type = "";
	Settings::getValue("main.type", type);
	if(!type.compare(0, 6, "binary") && !CommandSettings::get_onlyDataView()){ // type starts with binary -> remove all classes
		if(m_storage.size() > 2){
			Iterator it = m_storage.begin();
			++it; ++it; // go to the third element!
			for(;it != m_storage.end();){
				std::string name = it->first;
				for(unsigned int i = 0; i < it->second.size(); ++i){
					SAVE_DELETE(it->second[i]);
				}
				++it;
				m_storage.erase(name);
			}
		}
	}
	for(ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
		m_totalSize += it->second.size();
	}
	if(Settings::getDirectBoolValue("TotalStorage.normalizeData") && !didNormalizeStep){
		DataConverter::centerAndNormalizeData(m_storage, m_center, m_var);
	}
}

ClassPoint* TotalStorage::getDefaultEle(){
	return &m_defaultEle;
}

unsigned int TotalStorage::getTotalSize(){
	return m_totalSize;
}

unsigned int TotalStorage::getAmountOfClass(){
	return m_storage.size();
}

void TotalStorage::getOnlineStorageCopy(OnlineStorage<ClassPoint*>& storage){
	storage.resize(m_totalSize);
	for(ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
		for(ClassDataConstIterator itData = it->second.begin(); itData != it->second.end(); ++itData){
			storage.append(*itData);
		}
	}
}

void TotalStorage::getOnlineStorageCopyWithTest(OnlineStorage<ClassPoint*>& train,
		OnlineStorage<ClassPoint*>& test, const int amountOfPointsForTraining){
	int minValue = amountOfPointsForTraining / getAmountOfClass();
	for(ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
		minValue = std::min((int)it->second.size(), minValue);
	}
	std::vector<ClassPoint*> forTraining;
	std::vector<ClassPoint*> forTesting;
	for(ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
		int counter = 0;
		for(ClassDataConstIterator itData = it->second.begin(); itData != it->second.end(); ++itData){
			if(counter < minValue){
				forTraining.push_back(*itData);
			}else{
				forTesting.push_back(*itData);
			}
			++counter;
		}
	}
	printOnScreen("For training: " << forTraining.size());
	printOnScreen("For testing: " << forTesting.size());
	// to guarantee that the append block update is called which invokes the training
	train.append(forTraining);
	test.append(forTesting);
}

unsigned int TotalStorage::getSize(unsigned int classNr){
	Iterator it = m_storage.find(ClassKnowledge::getNameFor(classNr));
	if(it != m_storage.end()){
		return it->second.size();
	}
	return 0;
}

unsigned int TotalStorage::getSmallestClassSize(){
	unsigned int min = INT_MAX;
	for(ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
		min = std::min(min, (unsigned int) it->second.size());
	}
	return m_storage.size() != 0 ? min : 0;
}
