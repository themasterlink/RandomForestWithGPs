/*
 * OnlineStorage.h
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#ifndef DATA_ONLINESTORAGE_H_
#define DATA_ONLINESTORAGE_H_

#include "../Base/Observer.h"
#include <vector>

template<typename T>
class OnlineStorage : public Subject {
public:
	enum Event : unsigned int { // no class, because is used in to many places as an unsigned int
		APPEND = 0,
		APPENDBLOCK = 1,
		ERASE = 2,
		UNDEFINED = 3
	};

	using InternalStorage = std::vector<T>;
	using Iterator = typename InternalStorage::iterator;
	using ConstIterator = typename InternalStorage::const_iterator;

	OnlineStorage();

	OnlineStorage(OnlineStorage<T>& storage);

	void append(const T& data);

	void append(const std::vector<T>& data);

	void appendUnique(const std::vector<T>& data);

	void append(const OnlineStorage& storage);

	void remove(const Iterator& it);

	void resize(const unsigned int size);

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

	// always contains the index to last element before the last
	// append call was executed, in the beginning it is zero
	unsigned int m_lastUpdateIndex;
};

#define __INCLUDE_ONLINESTORAGE
#include "OnlineStorage_i.h"
#undef __INCLUDE_ONLINESTORAGE

#endif /* DATA_ONLINESTORAGE_H_ */
