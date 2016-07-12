/*
 * ThreadSafeOutput.cc
 *
 *  Created on: 12.07.2016
 *      Author: Max
 */

#include "ThreadSafeOutput.h"

ThreadSafeOutput::ThreadSafeOutput(std::ostream& stream): m_stream(stream) {
	// TODO Auto-generated constructor stub

}

ThreadSafeOutput::~ThreadSafeOutput() {
	// TODO Auto-generated destructor stub
}

void ThreadSafeOutput::print(const std::string& text){
	m_mutex.lock();
	m_stream << text << std::endl;
	m_mutex.unlock();
}

void ThreadSafeOutput::print(const std::string& text, const char* color){
	m_mutex.lock();
	m_stream << color << text << RESET << std::endl;
	m_mutex.unlock();
}
