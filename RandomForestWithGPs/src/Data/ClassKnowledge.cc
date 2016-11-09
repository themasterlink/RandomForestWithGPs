/*
 * ClassKnowledge.cc
 *
 *  Created on: 14.10.2016
 *      Author: Max
 */

#include "ClassKnowledge.h"
#include "../Utility/Util.h"

std::vector<std::string> ClassKnowledge::m_names;
unsigned int ClassKnowledge::m_amountOfDims(0);

ClassKnowledge::ClassKnowledge() {
}

ClassKnowledge::~ClassKnowledge() {
}

void ClassKnowledge::setNameFor(const std::string& name, unsigned int nr){
	if(nr <= m_names.size()){
		if(nr == m_names.size()){
			m_names.push_back(name);
		}else{ // change of name
			m_names[nr] = name;
		}
	}else{
		printError("The nr and the amount of the names does not correspond! Nr: " << nr << ", amount: " << m_names.size() << "!");
	}
}

std::string ClassKnowledge::getNameFor(unsigned int nr){
	if(nr < m_names.size()){
		return m_names[nr];
	}
	printError("This number has no name: " << nr << "!");
	return "undefined";
}

unsigned int ClassKnowledge::amountOfClasses(){
	return m_names.size();
}

unsigned int ClassKnowledge::amountOfDims(){
	return m_amountOfDims;
}

void ClassKnowledge::setAmountOfDims(unsigned int value){
	m_amountOfDims = value;
}
