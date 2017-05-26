/*
 * ClassKnowledge.cc
 *
 *  Created on: 14.10.2016
 *      Author: Max
 */

#include "ClassKnowledge.h"
#include "../Utility/Util.h"

ClassKnowledge::LabelNameMap ClassKnowledge::m_names;
unsigned int ClassKnowledge::m_amountOfDims(0);
boost::mutex ClassKnowledge::m_mutex;

ClassKnowledge::ClassKnowledge() {
}

ClassKnowledge::~ClassKnowledge() {
}

void ClassKnowledge::init(){
	m_names.clear();
	m_amountOfDims = 0;
	m_names.emplace(UNDEF_CLASS_LABEL, "undefined");
}

void ClassKnowledge::setNameFor(const std::string& name, unsigned int nr){
	m_mutex.lock();
	auto it = m_names.find(nr);
	if(it != m_names.end()){
		if(it->second == name){
			printError("This class was already added");
		}else{
			printError("This class was already added with another name!");
		}
	}else{
		m_names.emplace(nr, name);
	}
	if(nr >= UNDEF_CLASS_LABEL){
		printError("The amount of classes exceeds the amount of supported classes: " << UNDEF_CLASS_LABEL);
		sleepFor(10);
		m_mutex.unlock();
		exit(0);
	}
	m_mutex.unlock();
}

std::string ClassKnowledge::getNameFor(unsigned int nr){
	m_mutex.lock();
	auto it = m_names.find(nr);
	if(it != m_names.end()){
		const auto ret = it->second;
		m_mutex.unlock();
		return ret;
	}else{
		printError("This number has no name: " << nr << "!");
		const auto ret = m_names.find(UNDEF_CLASS_LABEL)->second;
		m_mutex.unlock();
		return ret;
	}
}

unsigned int ClassKnowledge::amountOfClasses(){
	m_mutex.lock();
	const auto size = (unsigned int) (m_names.size() - 1);// for default class!
	m_mutex.unlock();
	return size;
}

unsigned int ClassKnowledge::amountOfDims(){
	return m_amountOfDims;
}

void ClassKnowledge::setAmountOfDims(unsigned int value){
	m_mutex.lock();
	m_amountOfDims = value;
	m_mutex.unlock();
}

bool ClassKnowledge::hasClassName(const unsigned int nr){
	m_mutex.lock();
	const bool exists = m_names.end() != m_names.find(nr);
	m_mutex.unlock();
	return exists;
}
