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
#include <vector>

template<typename T>
class PoolInfo {
public:
	PoolInfo();

	std::vector<Real>& getPerformancesRef(){ return m_performance; };

	void changeAmountOfClasses(const unsigned int amountOfClasses);

	void setMaxNumberOfSavedPoints(const unsigned int maxNr);

	bool checkIfPointShouldBeAdded(const T& data);

private:
	std::vector<unsigned int> m_desiredSizes;
	std::vector<unsigned int> m_currentSizes;
	std::vector<Real> m_performance;
	unsigned int m_totalAmountOfSavedPoints;
	unsigned int m_amountOfPointsPerClass;
};

template<typename T>
class OnlineStorage: public Subject, public Observer {
public:
	enum Event: unsigned int { // no class, because is used in to many places as an unsigned int
		APPEND = 0,
		APPENDBLOCK = 1,
		ERASE = 2,
		UNDEFINED = 3
	};

	enum class StorageMode {
		POOL,
		NORMAL
	};

	using InternalStorage = std::vector<T>;
	using MultiClassInternalStorage = std::vector<InternalStorage>;
	using Iterator = typename InternalStorage::iterator;
	using ConstIterator = typename InternalStorage::const_iterator;

	OnlineStorage();

	OnlineStorage(OnlineStorage<T>& storage);

	void setStorageModeToPoolBase();

	void update(Subject* caller, unsigned int event);

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

	virtual ~OnlineStorage();

private:
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
