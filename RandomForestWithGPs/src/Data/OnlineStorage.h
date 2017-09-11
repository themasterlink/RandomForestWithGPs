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

/**
 * \brief The pool information of an online storage, saves the current amount of points for each class and the desired
 * 		size, according to the performance
 * \tparam T the element type, which is stored in the online storage
 */
template<typename T>
class PoolInfo : public Observer {
public:
	/**
	 * \brief Init it with the amount of class from the ClassKnowledge
	 */
	PoolInfo();

	/**
	 * \brief Get a reference to the performance array, which consist out of AvgNumbers
	 * \return
	 */
	std::vector<AvgNumber>& getPerformancesRef(){ return m_performance; };

	/**
	 * \brief Change the amount of used classes to the given value
	 * \param amountOfClasses the value for the amount of desired classes
	 */
	void changeAmountOfClasses(const unsigned int amountOfClasses);

	/**
	 * \brief Set the maximum number of saved points, set the maximum size of the pool
	 * \param maxNr maximum value for the size of the pool
	 */
	void setMaxNumberOfSavedPoints(const unsigned int maxNr);

	/**
	 * \brief Add a point of a given class to the current occupation of the storage.
	 * 		A check should be performed before.
	 * \param classNr given class
	 */
	void addPointToClass(const unsigned int classNr);

	/**
	 * \brief Update the desired sizes accordingly to the performance and the maximum set pool size,
	 * 			this function is called by the update method.
	 */
	void updateAccordingToPerformance();

	/**
	 * \brief Check if the point should be added to the storage (only check no adding is performed)
	 * \param data the point which should be checked
	 * \return true if there is enough space for the given points class
	 */
	bool checkIfPointShouldBeAdded(const T& data);

	/**
	 * \brief Get the class where points should be removed, returns UNDEF_CLASS_LABEL if no class should be removed
	 * \return The class which has too many points at the moment
	 */
	unsigned int getClassWherePointShouldBeRemoved();

	/**
	 * \brief Remove a certain amount of points from the occupied counter of a class
	 * \param classNr the class nr of where the points should be removed
	 * \param amount of points to be removed
	 */
	void removePointsFromClass(unsigned int classNr, unsigned int amount);

	/**
	 * \brief Get the difference between the occupation and the desired size for the given class
	 * \param classNr where the difference should be calculated
	 * \return the difference between the occupation and the desired size for the given class
	 */
	unsigned int getDifferenceForClass(const unsigned int classNr);

	/**
	 * \brief Update the desired sizes accordingly to the the performance, calls: updateAccordingToPerformance()
	 * \param caller which called this function
	 * \param event which started the update
	 */
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

	/**
	 * \brief Update of the Online Storage is only called if the class amount is changed (only relevant if pool is used)
	 * \param caller which evoced the call
	 * \param event which was called
	 */
	void update(Subject* caller, unsigned int event) override;

	/**
	 * \brief Append a new data point to the online storage (will cause an update of all observers)
	 * \param data a new data point
	 */
	void append(const T& data);

	/**
	 * \brief Append a new vector of data points to the online storage (will cause an update of all observers)
	 * \param data a new data vector
	 */
	void append(const std::vector<T>& data);

	/**
	 * \brief Append a new vector of data points to the online storage (will cause an update of all observers).
	 * 		Checks for every data point if this point is already in the storage.
	 * \param data a new data vector
	 */
	void appendUnique(const std::vector<T>& data);

	/**
	 * \brief Append the storage to the data points of the online storage (will cause an update of all observers)
	 * \param storage a new online storage
	 */
	void append(const OnlineStorage& storage);

	/**
	 * \brief Remove iterator of the storage, should not be done in an update step (RF)
	 * 		Implementation is not done for all Learners
	 * \param it a iterator
	 */
	void remove(const Iterator& it);

	/**
	 * \brief Access the element with the index, no boundary check!
	 * \param index of the element
	 * \return the element at index
	 */
	T& operator[](int index);

	/**
	 * \brief Access the const element with the index, no boundary check!
	 * \param index of the element
	 * \return the const element at index
	 */
	const T& operator[](int element) const;

	/**
	 * \brief Get a reference to the storage
	 * \return A reference to the storage
	 */
	InternalStorage& storage();

	/**
	 * \brief Get a reference to the const storage
	 * \return A const reference to the storage
	 */
	const InternalStorage& storage() const;

	/**
	 * \brief Return a iterator the begin of the storage
	 * \return Return a iterator
	 */
	Iterator begin();

	/**
	 * \brief Return a const iterator the end of the storage
	 * \return Return a const iterator
	 */
	Iterator end();

	/**
	 * \brief Return a const iterator the begin of the storage
	 * \return Return a const iterator
	 */
	ConstIterator begin() const;

	/**
	 * \brief Return a const iterator the end of the storage
	 * \return Return a const iterator
	 */
	ConstIterator end() const;

	/**
	 * \brief Return a const iterator the begin of the storage
	 * \return Return a const iterator
	 */
	ConstIterator cbegin() const;

	/**
	 * \brief Return a const iterator the end of the storage
	 * \return Return a const iterator
	 */
	ConstIterator cend() const;

	/**
	 * \brief Return a reference to the first element, check before if storage is empty or not!
	 * \return A reference to the first element
	 */
	T& first();

	/**
	 * \brief Return a reference to the last element, check before if storage is empty or not!
	 * \return A reference to the last element
	 */
	T& last();

	/**
	 * \brief Return the size of the current storage (is also the size of the pool)
	 * \return Size of the storage
	 */
	unsigned int size() const{ return (unsigned int) m_internal.size(); };

	/**
	 * \brief Return the dimension of the vectors stored in this storage
	 * \return The dimension of the vector
	 */
	unsigned int dim() const;

	/**
	 * \brief Get the start index of the last update step
	 * \return The index of the last update step
	 */
	unsigned int getLastUpdateIndex() const;

	/**
	 * \brief Get the amount of new points in the storage
	 * \return Get a amount of new points in the storage
	 */
	unsigned int getAmountOfNew() const;

	/**
	 * \brief Get the class type for the subject and observer
	 * \return The ClassTypeSubject::OnlineStorage
	 */
	ClassTypeSubject classType() const override;

	/**
	 * \brief Get a reference to the pool
	 * \return A reference to the pool
	 */
	PoolInfo<T>& getPoolInfoRef(){ return m_poolInfo; };

	/**
	 * \brief Return true if the pool mode is used
	 * \return true if the pool mode is used
	 */
	const bool isInPoolMode() const { return m_storageMode == StorageMode::POOL; };

	virtual ~OnlineStorage();

private:

	/**
	 * \brief Append the data vector to the storage, parameter determines if a check should be performed if the point
	 * 		is already in the storage
	 * \param data to be added to the storage
	 * \param shouldBeAddedUnique if the points should be checked before they get added
	 */
	void appendInternal(const std::vector<T>& data, const bool shouldBeAddedUnique);

	/**
	 * \brief The internal storage of the online storage
	 */
	InternalStorage m_internal;

	/**
	 * \brief The multi internal storage only used in the pool mode, is always convert to a internal storage after an append call
	 */
	MultiClassInternalStorage m_multiInternal;

	/**
	 * \brief The storage mode of the online storage (Pool or Normal Mode)
	 */
	StorageMode m_storageMode;

	/**
	 * \brief always contains the index to last element before the last append call was executed,
	 * 		in the beginning it is zero
	 */
	unsigned int m_lastUpdateIndex;

	/**
	 * \brief The pool, which is safes the performance for the classes and caluculates the current sizes
	 */
	PoolInfo<T> m_poolInfo;

	/**
	 * \brief copies the different pools in the internal storage to get fast access times during training
	 */
	void copyMultiInternalInInternal();

	/**
	 * \brief Check if the point should be added, only used for the pool mode (if there is enough space for its class)
	 * \param data the point which should be checked
	 * \return true if the point can be added
	 */
	bool checkIfPointShouldBeAdded(const T& data);
};

#define __INCLUDE_ONLINESTORAGE

#include "OnlineStorage_i.h"

#undef __INCLUDE_ONLINESTORAGE

#endif /* DATA_ONLINESTORAGE_H_ */
