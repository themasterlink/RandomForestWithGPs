/*
 * OnlineRandomForestIVMs.cc
 *
 *  Created on: 17.11.2016
 *      Author: Max
 */

#include "OnlineRandomForestIVMs.h"

OnlineRandomForestIVMs::OnlineRandomForestIVMs(OnlineStorage<ClassPoint*>& storage, const int maxDepth, const int amountOfUsedClasses):
	m_storage(storage),
	m_orf(storage, maxDepth, amountOfUsedClasses),
	m_amountOfUsedClasses(amountOfUsedClasses),
	m_firstTrainedDone(false){
	// removes orf to avoid that the update is called directly to the orf
	m_orf.getStorageRef().deattach(&m_orf);
	// instead call the update on the
	m_storage.attach(this);


}

OnlineRandomForestIVMs::~OnlineRandomForestIVMs() {
}

void OnlineRandomForestIVMs::update(Subject* caller, unsigned int event){
	switch(event){
	case OnlineStorage<ClassPoint*>::APPEND:{
		printError("This is not implemented yet!");
		break;
	}
	case OnlineStorage<ClassPoint*>::APPENDBLOCK:{
		update();
		break;
	}
	case OnlineStorage<ClassPoint*>::ERASE:{
		printError("This update type is not supported here!");
		break;
	}
	default: {
		printError("This update type is not supported here!");
		break;
	}
	}
}

void OnlineRandomForestIVMs::update(){
	if(!m_firstTrainedDone){
		m_orf.update(&m_storage, OnlineStorage<ClassPoint*>::APPENDBLOCK);
	}else{
		printError("Not implemented yet!");
	}

}

int OnlineRandomForestIVMs::predict(const DataPoint& point) const{
	printError("This function is not implemented");
	return -1;
}

void OnlineRandomForestIVMs::predictData(const Data& points, Labels& labels) const{
	printError("This function is not implemented");
}

void OnlineRandomForestIVMs::predictData(const Data& points, Labels& labels, std::vector< std::vector<double> >& probabilities) const{
	printError("This function is not implemented");
}

int OnlineRandomForestIVMs::amountOfClasses() const{
	return m_amountOfUsedClasses;
}
