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

typedef typename std::vector<ClassPoint*> ClassData;

typedef typename ClassData::iterator ClassDataIterator;

typedef typename ClassData::const_iterator ClassDataConstIterator;

#endif /* DATA_CLASSDATA_H_ */
