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

/**
 * \brief The total storage, which contains all points, all points are really saved in here.
 * 	Deleting points here might delete, leads to seg faults during the training
 * 	This class is a singeton, there can not be more than one instance of the TotalStorage
 */
class TotalStorage : public Singleton<TotalStorage> {

	friend class Singleton<TotalStorage>;

public:
	/**
	 * \brief The two different data set modes are available:
	 * 		WHOLE: the data set consist out of the training and the test set
	 * 		SEPARATE: there are two different sets, one for training and one for the test set
	 */
	enum class DataSetMode {
		WHOLE = 0,
		SEPARATE = 1
	};

	/**
	 * \brief Internal storage contains all data points, they are sorted after their classes
	 */
	using InternalStorage = DataSets;
	/**
	 * \brief A iterator for the internal storage
	 */
	using Iterator = DataSetsIterator;
	/**
	 * \brief A const iterator for the internal storage
	 */
	using ConstIterator = DataSetsConstIterator;

	/**
	 * \brief Read the data from the predefined files in the settings, depend on the m_dataSetMode
	 * \param amountOfData how many points are read
	 */
	void readData(const int amountOfData);

	/**
	 * \brief Return the total amount of points in the internal storage
	 * \return The amount of total points
	 */
	unsigned int getTotalSize();

	/**
	 * \brief Return the amount of classes
	 * \return the amount of classes
	 */
	unsigned int getAmountOfClass();

	/**
	 * \brief Return the size of the smallest class
	 * \return the size of the smallest class
	 */
	unsigned int getSmallestClassSize();

	/**
	 * \brief Copy the data in the training and the test set
	 * 		For the whole mode: the amountOfPointsForTraining are used to determine the size of the training set
	 * 				The size may vary if not all classes have the same size (it depends on the minimum amount of a class)
	 * 		For the separate mode: the training and test data are just copied
	 * 		At the end of the function the append is called, which starts a training to the online storage
	 * \param train the set which is later used for training
	 * \param test the set which is later used for testing
	 * \param amountOfPointsForTraining The amount of points, which are used if the whole mode is used
	 */
	void getOnlineStorageCopyWithTest(OnlineStorage<LabeledVectorX*>& train, OnlineStorage<LabeledVectorX*>& test,
									  const int amountOfPointsForTraining);

	/**
	 * \brief Copy the data in the training and the test set
	 * 		For the whole mode: the amountOfPointsForTraining are used to determine the size of the training set
	 * 				The size may vary if not all classes have the same size (it depends on the minimum amount of a class)
	 * 		For the separate mode: the training and test data are just copied
	 * \param train the set which is later used for training
	 * \param test the set which is later used for testing
	 * \param amountOfPointsForTraining The amount of points, which are used if the whole mode is used
	 */
	void getLabeledDataCopyWithTest(LabeledData& train, LabeledData& test, const int amountOfPointsForTraining);

	/**
	 * \brief Get the points, which were removed from the training and test set during the initial reading process
	 * 			Works only for separation, not implemented for whole
	 * \param train the set which is later used for training
	 * \param test the set which is later used for testing
	 */
	void getRemovedOnlineStorageCopyWithTest(OnlineStorage<LabeledVectorX*>& train,
											 OnlineStorage<LabeledVectorX*>& test);

	/**
	 * \brief Copy the data in the training and the test set
	 * 		For the whole mode: the amountOfPointsForTraining are used to determine the size of the training set
	 * 				The size may vary if not all classes have the same size (it depends on the minimum amount of a class)
	 * 		For the separate mode: the training and test data are just copied
	 * 		At the end of the function the append is called, which starts a training to the online storage
	 * 		Each set is appended after each other to the vector
	 * \param trains the vectors which contain the training splits
	 * \param test the set which is later used for testing
	 */
	void getOnlineStorageCopySplitsWithTest(std::vector<OnlineStorage<LabeledVectorX*> >& trains,
											OnlineStorage<LabeledVectorX*>& test);

	/**
	 * \brief Return a pointer to the validation set, returns nullptr if no validation set is used
	 * \return pointer to validation set
	 */
	LabeledData* getValidationSet();

	/**
	 * \brief Return the current data set mode
	 * \return the current data set mode
	 */
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
