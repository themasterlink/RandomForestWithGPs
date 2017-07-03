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
#include "../Base/Types.h"
#include "../Base/Observer.h"

class ClassKnowledge {

SINGELTON_MACRO(ClassKnowledge);
	
public:
	class Caller: public Subject {
	public:
		enum Event {
			NEW_CLASS = 500,
			UNDEFINED = 501
		};

		ClassTypeSubject classType() const{ return ClassTypeSubject::CLASSKNOWLEDGE; };
	};

	using LabelNamePair = std::pair<unsigned int, std::string>;

	using LabelNameMap = std::map<unsigned int, std::string > ;

	using LabelNameMapIterator = LabelNameMap::iterator;

	void setNameFor(const std::string& name, unsigned int nr);

	std::string getNameFor(unsigned int nr);

	unsigned int amountOfClasses();

	unsigned int amountOfDims();

	void setAmountOfDims(unsigned int value);

	bool hasClassName(const unsigned int nr);

	void init();

	void attach(Observer* obj);

	void deattach(Observer* obj);

private:

	Caller m_caller;

	LabelNameMap m_names;

	unsigned int m_amountOfDims;

	Mutex m_mutex;

};

#endif /* DATA_CLASSKNOWLEDGE_H_ */
