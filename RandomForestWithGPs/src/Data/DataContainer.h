/*
 * DataContainer.h
 *
 *  Created on: 21.07.2016
 *      Author: Max
 */

#ifndef DATA_DATACONTAINER_H_
#define DATA_DATACONTAINER_H_

#include "Data.h"

class DataContainer {
public:
	DataContainer();
	virtual ~DataContainer();

	void fillWith(const DataSets& dataSets);

	Data data;
	Labels labels;
	std::vector<std::string> namesOfClasses;
	int amountOfPoints;
	int amountOfClasses;
};

#endif /* DATA_DATACONTAINER_H_ */
