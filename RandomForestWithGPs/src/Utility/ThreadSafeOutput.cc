/*
 * ThreadSafeOutput.cc
 *
 *  Created on: 12.07.2016
 *      Author: Max
 */

#include "ThreadSafeOutput.h"

ThreadSafeOutput::ThreadSafeOutput(std::ostream& stream): m_stream(stream), m_change(false) {
	// TODO Auto-generated constructor stub

}

ThreadSafeOutput::~ThreadSafeOutput() {
	// TODO Auto-generated destructor stub
}

void ThreadSafeOutput::print(const std::string& text){
	lockStatementWith(m_stream << text << std::endl, m_mutex);
}

void ThreadSafeOutput::printSwitchingColor(const std::string& text){
	m_mutex.lock();
	if(m_change){
		m_stream << RED;
	}else{
		m_stream << CYAN;
	}
	m_change = !m_change;
	m_stream << text << RESET << std::endl;
	m_mutex.unlock();
}

void ThreadSafeOutput::printInColor(const std::string& text, const char* color){
	lockStatementWith(m_stream << color << text << RESET << std::endl, m_mutex);
}
