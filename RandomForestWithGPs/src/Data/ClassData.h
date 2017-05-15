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

using ClassData = std::vector<ClassPoint*>;

using ClassDataIterator = ClassData::iterator;

using ClassDataConstIterator = ClassData::const_iterator;

#endif /* DATA_CLASSDATA_H_ */
