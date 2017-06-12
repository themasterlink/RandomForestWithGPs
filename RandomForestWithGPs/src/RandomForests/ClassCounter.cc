//
// Created by denn_ma on 6/12/17.
//

#include "ClassCounter.h"

unsigned int ClassCounter::operator[](unsigned int classNr) const {
	auto it = m_classCounter.find(classNr);
	if(it != m_classCounter.end()){
		return it->second;
	}else{
		return 0;
	}
}

void ClassCounter::increment(unsigned int classNr){
	auto it = m_classCounter.find(classNr);
	if(it != m_classCounter.end()){
		++(it->second);
	}else{
		m_classCounter.emplace(classNr, 0);
	}
}

void ClassCounter::decrement(unsigned int classNr){
	auto it = m_classCounter.find(classNr);
	if(it != m_classCounter.end()){
		--(it->second);
		if(it->second <= 0){ // avoids collecting of useless classes
			m_classCounter.erase(classNr);
		}
	}else{
		printError("This class is not used before: " << classNr);
	};
}

unsigned int ClassCounter::argMax(){
	unsigned first = UNDEF_CLASS_LABEL;
	unsigned int biggestVal = 0;
	for(auto& it : m_classCounter){
		if(it.second >= biggestVal){
			first = it.first;
			biggestVal = it.second;
		}
	}
	return first;
}

unsigned int ClassCounter::incrementWithChange(unsigned int classNr, unsigned int& oldMaxClass){
	oldMaxClass = argMax();
	auto it = m_classCounter.find(classNr);
	if(it != m_classCounter.end()){
		++(it->second);
		if(classNr != oldMaxClass){
			if(it->second > m_classCounter.find(oldMaxClass)->second){
				return classNr;
			}
		}
	}else{
		m_classCounter.emplace(classNr, 0);
	}
	return oldMaxClass;
}