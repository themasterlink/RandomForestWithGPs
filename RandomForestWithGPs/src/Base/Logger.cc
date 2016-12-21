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
std::map<std::string, std::string> Logger::m_specialLines;

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
	write();
	m_mutex.unlock();
}

void Logger::write(){
	// not locked!
	std::fstream file;
	file.open(m_filePath, std::fstream::out | std::fstream::trunc);
	file.write(m_text.c_str(), m_text.length());
	for(std::map<std::string, std::string>::iterator it = m_specialLines.begin(); it != m_specialLines.end(); ++it){
		if(!(it->first == "Error" || it->first == "Warning")){
			file << it->first << "\n";
			file.write(it->second.c_str(), it->second.length());
		}
	}
	for(auto name : {"Warning", "Error"}){
		std::map<std::string, std::string>::iterator itOther = m_specialLines.find(name);
		if(itOther != m_specialLines.end()){
			file << itOther->first << "\n";
			file.write(itOther->second.c_str(), itOther->second.length());
		}
	}
	file.close();
	m_needToWrite = false;
}

void Logger::run(){
	while(m_init){
		m_mutex.lock();
		if(m_needToWrite){ // write only if something changed
			write();
		}
		m_mutex.unlock();
		usleep(m_timeToSleep * 1e6);
	}
}

void Logger::addNormalLineToFile(const std::string& line){
	if(m_init){
		m_mutex.lock();
		m_needToWrite = true;
		m_text += line + "\n";
		m_mutex.unlock();
	}
}

void Logger::addSpecialLineToFile(const std::string& line, const std::string& identifier){
	if(m_init){
		m_mutex.lock();
		std::map<std::string, std::string>::iterator it = m_specialLines.find(identifier);
		if(it != m_specialLines.end()){
			it->second += ("\t" + line + "\n");
		}else{
			const std::string input = "\t" + line + "\n";
			m_specialLines.insert(std::pair<std::string, std::string>(identifier, input));
		}
		m_needToWrite = true;
		m_mutex.unlock();
	}
}
