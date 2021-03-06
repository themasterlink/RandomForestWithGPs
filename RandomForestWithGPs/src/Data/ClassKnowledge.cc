/*
 * ClassKnowledge.cc
 *
 *  Created on: 14.10.2016
 *      Author: Max
 */

#include "ClassKnowledge.h"
#include "../Utility/Util.h"

ClassKnowledge::ClassKnowledge(): m_amountOfDims(0){
}

void ClassKnowledge::init(){
	m_names.clear();
	m_amountOfDims = 0;
	m_names.emplace(UNDEF_CLASS_LABEL, "undefined");
	m_caller.notify(Caller::NEW_CLASS);
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
		m_mutex.unlock();
		m_caller.notify(Caller::NEW_CLASS); // ensures that all storages are notified over the change
		return;
	}
	if(nr >= UNDEF_CLASS_LABEL){
		m_mutex.unlock();
		printErrorAndQuit("The amount of classes exceeds the amount of supported classes: " << UNDEF_CLASS_LABEL);
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
	lockStatementWithSave((unsigned int) (m_names.size() - 1), const auto size, m_mutex);
	return size;
}

unsigned int ClassKnowledge::amountOfDims(){
	return m_amountOfDims;
}

void ClassKnowledge::setAmountOfDims(unsigned int value){
	lockStatementWith(m_amountOfDims = value, m_mutex);
}

bool ClassKnowledge::hasClassName(const unsigned int nr){
	lockStatementWithSave(m_names.end() != m_names.find(nr), const bool exists, m_mutex);
	return exists;
}

void ClassKnowledge::attach(Observer* obj){
	lockStatementWith(m_caller.attach(obj), m_mutex);
}

void ClassKnowledge::deattach(Observer* obj){
	lockStatementWith(m_caller.detach(obj), m_mutex);
}