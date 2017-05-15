

#ifndef DATA_DATASETS_H_
#define DATA_DATASETS_H_


#include "ClassData.h"
#include <map>

using DataSets = std::map< std::string, ClassData>;

using DataSetsIterator = DataSets::iterator;

using DataSetsConstIterator = DataSets::const_iterator;

using DataSetPair = std::pair<std::string, ClassData >;

#endif
