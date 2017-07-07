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
#include "../Base/Singleton.h"

class TotalStorage : public Singleton<TotalStorage> {

	friend class Singleton<TotalStorage>;

public:

	enum class DataSetMode {
		WHOLE = 0,
		SEPERATE = 1
	};

	using InternalStorage = DataSets;
	using Iterator = DataSetsIterator;
	using ConstIterator = DataSetsConstIterator;

	void readData(const int amountOfData);

	unsigned int getTotalSize();

	unsigned int getAmountOfClass();

	unsigned int getSmallestClassSize();

	void getOnlineStorageCopyWithTest(OnlineStorage<LabeledVectorX*>& train, OnlineStorage<LabeledVectorX*>& test,
									  const int amountOfPointsForTraining);

	void getLabeledDataCopyWithTest(LabeledData& train, LabeledData& test, const int amountOfPointsForTraining);

	void
	getRemovedOnlineStorageCopyWithTest(OnlineStorage<LabeledVectorX*>& train, OnlineStorage<LabeledVectorX*>& test);;

	void getOnlineStorageCopySplitsWithTest(std::vector<OnlineStorage<LabeledVectorX*> >& trains,
											OnlineStorage<LabeledVectorX*>& test);

	LabeledData* getValidationSet();

	DataSetMode getDataSetMode(){ return m_dataSetMode; };

private:

	VectorX m_center;

	VectorX m_var;

	InternalStorage m_storage;

	LabeledData m_trainSet;

	LabeledData m_testSet;

	LabeledData m_validationSet;

	LabeledData m_removeFromTrainSet;

	LabeledData m_removeFromTestSet;

	DataSetMode m_dataSetMode;

	unsigned int m_totalSize;

	TotalStorage();
	~TotalStorage() = default;
};

#endif /* DATA_TOTALSTORAGE_H_ */
