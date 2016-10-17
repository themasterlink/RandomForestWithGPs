/*
 * TotalStorage.h
 *
 *  Created on: 14.10.2016
 *      Author: Max
 */

#ifndef DATA_TOTALSTORAGE_H_
#define DATA_TOTALSTORAGE_H_

#include <vector>
#include "ClassPoint.h"
#include "DataSets.h"

class TotalStorage {
public:

	typedef DataSets InternalStorage;
	typedef DataSetsIterator Iterator;
	typedef DataSetsConstIterator ConstIterator;

	static ClassPoint* getData(unsigned int classNr, unsigned int elementNr);

	static ClassPoint* getDefaultEle();

	static void readData(const std::string& folderLocation, const int amountOfData);

	static unsigned int getTotalSize();

	static unsigned int getSize(unsigned int classNr);

private:

	static InternalStorage m_storage;

	static ClassPoint m_defaultEle;

	static unsigned int m_totalSize;

	TotalStorage();
	virtual ~TotalStorage();
};

#endif /* DATA_TOTALSTORAGE_H_ */
