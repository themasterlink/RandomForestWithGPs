/*
 * ClassKnowledge.h
 *
 *  Created on: 14.10.2016
 *      Author: Max
 */

#ifndef DATA_CLASSKNOWLEDGE_H_
#define DATA_CLASSKNOWLEDGE_H_

#include <string>
#include <map>
#include <boost/thread.hpp>

class ClassKnowledge {
public:

	using LabelNamePair = std::pair<unsigned int, std::string>;

	using LabelNameMap = std::map<unsigned int, std::string > ;

	using LabelNameMapIterator = LabelNameMap::iterator;

	static void setNameFor(const std::string& name, unsigned int nr);

	static std::string getNameFor(unsigned int nr);

	static unsigned int amountOfClasses();

	static unsigned int amountOfDims();

	static void setAmountOfDims(unsigned int value);

	static bool hasClassName(const unsigned int nr);

	static void init();

private:

	static LabelNameMap m_names;

	static unsigned int m_amountOfDims;

	static boost::mutex m_mutex;

	ClassKnowledge();
	virtual ~ClassKnowledge();
};

#endif /* DATA_CLASSKNOWLEDGE_H_ */
