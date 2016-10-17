/*
 * ClassKnowledge.h
 *
 *  Created on: 14.10.2016
 *      Author: Max
 */

#ifndef DATA_CLASSKNOWLEDGE_H_
#define DATA_CLASSKNOWLEDGE_H_

#include <string>
#include <vector>

class ClassKnowledge {
public:

	static void setNameFor(const std::string& name, unsigned int nr);

	static std::string getNameFor(unsigned int nr);

	static unsigned int amountOfClasses();

private:

	static std::vector<std::string> m_names;

	ClassKnowledge();
	virtual ~ClassKnowledge();
};

#endif /* DATA_CLASSKNOWLEDGE_H_ */
