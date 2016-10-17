/*
 * OnlineStorage.h
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#ifndef DATA_ONLINESTORAGE_H_
#define DATA_ONLINESTORAGE_H_

#include "../Base/Observer.h"

template<typename T>
class OnlineStorage : public Subject {
public:
	enum Event : unsigned int {
		APPEND = 0,
		APPENDBLOCK = 1,
		ERASE = 2,
		UNDEFINED = 3
	};

	typedef typename std::list<T> InternalStorage;
	typedef typename InternalStorage::iterator Iterator;
	typedef typename InternalStorage::const_iterator ConstIterator;

	OnlineStorage();

	void append(const T& data);

	void append(const OnlineStorage& storage);

	void remove(const Iterator& it);

	virtual ~OnlineStorage();

private:
	InternalStorage m_internal;

};

#define __INCLUDE_ONLINESTORAGE
#include "OnlineStorage_i.h"
#undef __INCLUDE_ONLINESTORAGE

#endif /* DATA_ONLINESTORAGE_H_ */
