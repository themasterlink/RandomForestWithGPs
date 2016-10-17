/*
 * ClassData.h
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#ifndef DATA_CLASSDATA_H_
#define DATA_CLASSDATA_H_

#include "ClassPoint.h"
#include <vector>

class ClassData : public std::vector<ClassPoint*> {
public:
	ClassData();

	ClassData(const int size);

	virtual ~ClassData();
};

typedef typename ClassData::iterator ClassDataIterator;

typedef typename ClassData::const_iterator ClassDataConstIterator;

#endif /* DATA_CLASSDATA_H_ */
