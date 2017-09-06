/*
 * OnlineStorage.h
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#ifndef DATA_ONLINESTORAGE_H_
#define DATA_ONLINESTORAGE_H_

#include "../Base/Observer.h"
#include "ClassKnowledge.h"
#include "../Base/BaseType.h"
#include "../Utility/AvgNumber.h"
#include <vector>

template<typename T>
class PoolInfo : public Observer {
public:
	PoolInfo();

	std::vector<AvgNumber>& getPerformancesRef(){ return m_performance; };

	void changeAmountOfClasses(const unsigned int amountOfClasses);

	void setMaxNumberOfSavedPoints(const unsigned int maxNr);

	void addPointToClass(const unsigned int classNr);

	void updateAccordingToPerformance();

	bool checkIfPointShouldBeAdded(const T& data);

	unsigned int getClassWherePointShouldBeRemoved();

	void removePointsFromClass(unsigned int classNr, unsigned int amount);

	unsigned int getDifferenceForClass(const unsigned int classNr);

	void update(Subject* caller, unsigned int event) override;

private:
	std::vector<unsigned int> m_desiredSizes;
	std::vector<unsigned int> m_currentSizes;
	std::vector<AvgNumber> m_performance;
	unsigned int m_totalAmountOfSavedPoints;
	unsigned int m_amountOfPointsPerClass;
};

/**
 * \brief An online storage, which can be updated with new points, during the execution.
 * 		Each update of the storage causes an event, which triggers all observer.
 * \tparam T specify the element which is stored in the storage
 */
template<typename T>
class OnlineStorage: public Subject, public Observer {
public:
	/**
	 * \brief all the events which are available for the update of the online storage.
	 * 		APPEND is only for one point, where as APPENDBLOCK means that more than one point was added at the same time
	 * 	UPDATE_POOL_ACCORDING_TO_PERFORMANCE, is used to update the pool
	 */
	enum Event: unsigned int { // no class, because is used in to many places as an unsigned int
		APPEND = 0,
		APPENDBLOCK = 1,
		ERASE = 2,
		UNDEFINED = 3,
		UPDATE_POOL_ACCORDING_TO_PERFORMANCE = 4
	};

	/**
	 * \brief The storage mode specifiy if all points are saved in a conventional storage or if a pool is used, where
	 * the class size depends on the performance.
	 */
	enum class StorageMode {
		POOL,
		NORMAL
	};

	/**
	 * \brief Internal Storage used to save the incoming data, a vector is used, because updates are more seldom than
	 * accesses to the vector, so this was the priority for the optimization
	 */
	using InternalStorage = std::vector<T>;
	/**
	 * \brief The internal class storage, which contains all points of a certain class, a vector is used, because
	 * updates are more seldom than accesses to the vector, so this was the priority for the optimization
	 */
	using ClassInternalStorage = std::vector<T>;
	/**
	 * \brief Each class gets its own class internal storage, which is after the update resaved in the internal storage
	 */
	using MultiClassInternalStorage = std::vector<ClassInternalStorage>;
	/**
	 * \brief the iterator over the internal storage
	 */
	using Iterator = typename InternalStorage::iterator;
	/**
	 * \brief the const iterator over the internal storage
	 */
	using ConstIterator = typename InternalStorage::const_iterator;

	/**
	 * \brief Std. constructor, inits a empty storage in NORMAL mode
	 */
	OnlineStorage();

	/**
	 * \brief Std. copy constructor, make a copy of the storage
	 */
	OnlineStorage(OnlineStorage<T>& storage);

	/**
	 * \brief Change the mode of the storage from NORMAL to POOL
	 */
	void setStorageModeToPoolBase();

	void update(Subject* caller, unsigned int event) override;

	void append(const T& data);

	void append(const std::vector<T>& data);

	void appendUnique(const std::vector<T>& data);

	void append(const OnlineStorage& storage);

	void remove(const Iterator& it);

	T& operator[](int element);

	const T& operator[](int element) const;

	InternalStorage& storage();

	const InternalStorage& storage() const;

	Iterator begin();

	Iterator end();

	ConstIterator begin() const;

	ConstIterator end() const;

	ConstIterator cbegin() const;

	ConstIterator cend() const;

	T& first();

	T& last();

	unsigned int size() const{ return (unsigned int) m_internal.size(); };

	unsigned int dim() const;

	unsigned int getLastUpdateIndex() const;

	unsigned int getAmountOfNew() const;

	ClassTypeSubject classType() const override;

	PoolInfo<T>& getPoolInfoRef(){ return m_poolInfo; };

	const bool isInPoolMode() const { return m_storageMode == StorageMode::POOL; };

	virtual ~OnlineStorage();

private:

	void appendInternal(const std::vector<T>& data, const bool shouldBeAddedUnique);

	InternalStorage m_internal;

	MultiClassInternalStorage m_multiInternal;

	StorageMode m_storageMode;

	// always contains the index to last element before the last
	// append call was executed, in the beginning it is zero
	unsigned int m_lastUpdateIndex;

	PoolInfo<T> m_poolInfo;

	// copies the different pools in the internal storage to get fast access times during training
	void copyMultiInternalInInternal();

	bool checkIfPointShouldBeAdded(const T& data);
};

#define __INCLUDE_ONLINESTORAGE

#include "OnlineStorage_i.h"

#undef __INCLUDE_ONLINESTORAGE

#endif /* DATA_ONLINESTORAGE_H_ */
