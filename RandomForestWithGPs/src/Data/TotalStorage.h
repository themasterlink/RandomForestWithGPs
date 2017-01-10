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
#include "OnlineStorage.h"

class TotalStorage {
public:

	enum Mode {
		WHOLE = 0,
		SEPERATE = 1
	};

	typedef DataSets InternalStorage;
	typedef DataSetsIterator Iterator;
	typedef DataSetsConstIterator ConstIterator;

	static ClassPoint* getData(unsigned int classNr, unsigned int elementNr);

	static ClassPoint* getDefaultEle();

	static void readData(const int amountOfData);

	static unsigned int getTotalSize();

	static unsigned int getAmountOfClass();

	static unsigned int getSize(unsigned int classNr);

	static unsigned int getSmallestClassSize();

	static void getOnlineStorageCopy(OnlineStorage<ClassPoint*>& storage);

	static void getOnlineStorageCopyWithTest(OnlineStorage<ClassPoint*>& train, OnlineStorage<ClassPoint*>& test, const int amountOfPointsForTraining);

	static InternalStorage& getStorage(){ return m_storage; };

private:

	static DataPoint m_center;

	static DataPoint m_var;

	static InternalStorage m_storage;

	static ClassData m_trainSet;

	static ClassData m_testSet;

	static ClassPoint m_defaultEle;

	static Mode m_mode;

	static unsigned int m_totalSize;

	TotalStorage();
	virtual ~TotalStorage();
};

#endif /* DATA_TOTALSTORAGE_H_ */
