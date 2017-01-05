/*
 * ClassKnowledge.cc
 *
 *  Created on: 14.10.2016
 *      Author: Max
 */

#include "ClassKnowledge.h"
#include "../Utility/Util.h"
#include "Data.h"

ClassKnowledge::LabelNameMap ClassKnowledge::m_names;
unsigned int ClassKnowledge::m_amountOfDims(0);

ClassKnowledge::ClassKnowledge() {
}

ClassKnowledge::~ClassKnowledge() {
}

void ClassKnowledge::init(){
	m_names.insert(LabelNamePair(UNDEF_CLASS_LABEL, "undefined"));
}

void ClassKnowledge::setNameFor(const std::string& name, unsigned int nr){
	LabelNameMapIterator it = m_names.find(nr);
	if(it != m_names.end()){
		if(it->second == name){
			printError("This class was already added");
		}else{
			printError("This class was already added with another name!");
		}
	}else{
		m_names.insert(LabelNamePair(nr, name));
	}
	if(nr >= UNDEF_CLASS_LABEL){
		printError("The amount of classes exceeds the amount of supported classes: " << UNDEF_CLASS_LABEL);
		sleep(10);
		exit(0);
	}
}

std::string ClassKnowledge::getNameFor(unsigned int nr){
	LabelNameMapIterator it = m_names.find(nr);
	if(it != m_names.end()){
		return it->second;
	}else{
		printError("This number has no name: " << nr << "!");
		return m_names.find(UNDEF_CLASS_LABEL)->second;
	}
}

unsigned int ClassKnowledge::amountOfClasses(){
	return m_names.size() - 1; // for default class!
}

unsigned int ClassKnowledge::amountOfDims(){
	return m_amountOfDims;
}

void ClassKnowledge::setAmountOfDims(unsigned int value){
	m_amountOfDims = value;
}

bool ClassKnowledge::hasClassName(const unsigned int nr){
	return m_names.end() != m_names.find(nr);
}
