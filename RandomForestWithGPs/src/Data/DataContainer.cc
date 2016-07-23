/*
 * DataContainer.cc
 *
 *  Created on: 21.07.2016
 *      Author: Max
 */

#include "DataContainer.h"

DataContainer::DataContainer():amountOfPoints(0),amountOfClasses(0) {
}

DataContainer::~DataContainer() {
}


void DataContainer::fillWith(const DataSets& dataSets){
	const int dim = dataSets.begin()->second[0].rows();
	// count total data points in dataset
	amountOfClasses = dataSets.size();
	amountOfPoints = 0;
	for(DataSets::const_iterator it = dataSets.begin(); it != dataSets.end(); ++it){
		amountOfPoints += it->second.size();
	}
	// copy all points in one Data field for training of the RF
	data.resize(amountOfPoints);
	labels.resize(amountOfPoints);
	namesOfClasses.resize(dataSets.size());
	int labelsCounter = 0, offset = 0;
	for(DataSets::const_iterator it = dataSets.begin(); it != dataSets.end(); ++it){
		namesOfClasses[labelsCounter] = it->first;
		for(int i = 0; i < it->second.size(); ++i){
			labels[offset + i] = labelsCounter;
			data[offset + i] = it->second[i];
		}
		offset += it->second.size();
		++labelsCounter;
	}
}
