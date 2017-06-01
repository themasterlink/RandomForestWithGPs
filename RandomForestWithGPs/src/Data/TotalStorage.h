/*
 * TotalStorage.h
 *
 *  Created on: 14.10.2016
 *      Author: Max
 */

#ifndef DATA_TOTALSTORAGE_H_
#define DATA_TOTALSTORAGE_H_

#include <vector>
#include "LabeledVectorX.h"
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

	static LabeledVectorX* getData(unsigned int classNr, unsigned int elementNr);

	static LabeledVectorX* getDefaultEle();

	static void readData(const int amountOfData);

	static unsigned int getTotalSize();

	static unsigned int getAmountOfClass();

	static unsigned int getSize(unsigned int classNr);

	static unsigned int getSmallestClassSize();

	static void getOnlineStorageCopy(OnlineStorage<LabeledVectorX*>& storage);

	static void getOnlineStorageCopyWithTest(OnlineStorage<LabeledVectorX*>& train, OnlineStorage<LabeledVectorX*>& test, const int amountOfPointsForTraining);

	static void getLabeledDataCopyWithTest(LabeledData& train, LabeledData& test, const int amountOfPointsForTraining);

	static void getRemovedOnlineStorageCopyWithTest(OnlineStorage<LabeledVectorX*>& train, OnlineStorage<LabeledVectorX*>& test);

	static InternalStorage& getStorage(){ return m_storage; };

	static void getOnlineStorageCopySplitsWithTest(std::vector<OnlineStorage<LabeledVectorX*> >& trains, OnlineStorage<LabeledVectorX*>& test);

	static Mode getMode(){ return m_mode; };

private:

	static VectorX m_center;

	static VectorX m_var;

	static InternalStorage m_storage;

	static LabeledData m_trainSet;

	static LabeledData m_testSet;

	static LabeledData m_removeFromTrainSet;

	static LabeledData m_removeFromTestSet;

	static LabeledVectorX m_defaultEle;

	static Mode m_mode;

	static unsigned int m_totalSize;

	TotalStorage();
	virtual ~TotalStorage();
};

#endif /* DATA_TOTALSTORAGE_H_ */
