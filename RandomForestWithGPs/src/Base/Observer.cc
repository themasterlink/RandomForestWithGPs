/*
 * Observer.cc
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#include "Observer.h"

Observer::Observer() {
}

Observer::~Observer() {
}

Subject::Subject(){
}

Subject::~Subject(){
}

void Subject::attach(Observer* obj){
	if(obj != nullptr){
		m_observers.push_back(obj);
	}
}

void Subject::deattach(Observer* obj){
	if(obj != nullptr){
		m_observers.remove(obj);
	}
}

void Subject::deattachAll(){
	m_observers.clear();
}

void Subject::notify(const unsigned int event){
	for(std::list<Observer*>::iterator it = m_observers.begin(); it != m_observers.end(); ++it){
		if((*it) != nullptr){
			(*it)->update(this, event);
		}
	}
}

unsigned int Subject::numberOfObservers() const{
	return m_observers.size();
}