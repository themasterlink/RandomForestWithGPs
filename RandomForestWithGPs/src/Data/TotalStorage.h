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

	enum class Mode {
		WHOLE = 0,
		SEPERATE = 1
	};

	using InternalStorage = DataSets;
	using Iterator = DataSetsIterator;
	using ConstIterator = DataSetsConstIterator;

	static ClassPoint* getData(unsigned int classNr, unsigned int elementNr);

	static ClassPoint* getDefaultEle();

	static void readData(const int amountOfData);

	static unsigned int getTotalSize();

	static unsigned int getAmountOfClass();

	static unsigned int getSize(unsigned int classNr);

	static unsigned int getSmallestClassSize();

	static void getOnlineStorageCopy(OnlineStorage<ClassPoint*>& storage);

	static void getOnlineStorageCopyWithTest(OnlineStorage<ClassPoint*>& train, OnlineStorage<ClassPoint*>& test, const int amountOfPointsForTraining);

	static void getRemovedOnlineStorageCopyWithTest(OnlineStorage<ClassPoint*>& train, OnlineStorage<ClassPoint*>& test);

	static InternalStorage& getStorage(){ return m_storage; };

	static void getOnlineStorageCopySplitsWithTest(std::vector<OnlineStorage<ClassPoint*> >& trains, OnlineStorage<ClassPoint*>& test);

	static Mode getMode(){ return m_mode; };

private:

	static DataPoint m_center;

	static DataPoint m_var;

	static InternalStorage m_storage;

	static ClassData m_trainSet;

	static ClassData m_testSet;

	static ClassData m_removeFromTrainSet;

	static ClassData m_removeFromTestSet;

	static ClassPoint m_defaultEle;

	static Mode m_mode;

	static unsigned int m_totalSize;

	TotalStorage();
	virtual ~TotalStorage();
};

#endif /* DATA_TOTALSTORAGE_H_ */
