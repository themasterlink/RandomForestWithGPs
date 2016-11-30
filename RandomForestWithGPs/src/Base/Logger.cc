/*
 * Logger.cc
 *
 *  Created on: 25.11.2016
 *      Author: Max
 */

#include "Logger.h"
#include "Settings.h"
#include "boost/date_time/posix_time/posix_time.hpp"

boost::mutex Logger::m_mutex;
boost::thread* Logger::m_ownThread(nullptr);
bool Logger::m_init(false);
bool Logger::m_needToWrite(false);
std::string Logger::m_text("");
std::string Logger::m_filePath("");
double Logger::m_timeToSleep(2.);

Logger::Logger() {
}

Logger::~Logger() {
}

void Logger::start(){
	m_init = Settings::getDirectBoolValue("Logger.useLogger");
	if(m_init){
		Settings::getValue("Logger.filePath", m_filePath);
		std::string mode;
		Settings::getValue("main.type", mode);
		m_text = "Online Random Forest with IVMs, mode: " + mode + "\n"; // Standart Information
		m_text += "Date: " + boost::posix_time::to_simple_string(boost::posix_time::second_clock::local_time()) + "\n";
		m_ownThread = new boost::thread(&Logger::run);
	}
}

void Logger::forcedWrite(){
	m_mutex.lock();
	std::fstream file;
	file.open(m_filePath, std::fstream::out | std::fstream::trunc);
	file.write(m_text.c_str(), m_text.length());
	file.close();
	m_needToWrite = false;
	m_mutex.unlock();
}

void Logger::run(){
	while(m_init){
		m_mutex.lock();
		if(m_needToWrite){ // write only if something changed
			std::fstream file;
			file.open(m_filePath, std::fstream::out | std::fstream::trunc);
			file.write(m_text.c_str(), m_text.length());
			file.close();
			m_needToWrite = false;
		}
		m_mutex.unlock();
		usleep(m_timeToSleep * 1e6);
	}
}

void Logger::addToFile(const std::string& line){
	if(m_init){
		m_mutex.lock();
		m_needToWrite = true;
		m_text += line + "\n";
		m_mutex.unlock();
	}
}
