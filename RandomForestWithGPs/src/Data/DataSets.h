

#ifndef DATA_DATASETS_H_
#define DATA_DATASETS_H_


#include "ClassData.h"
#include <map>

typedef typename std::map< std::string, ClassData> DataSets;

typedef typename DataSets::iterator DataSetsIterator;

typedef typename DataSets::const_iterator DataSetsConstIterator;

typedef typename std::pair<std::string, ClassData > DataSetPair;

#endif
