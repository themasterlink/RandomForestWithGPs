/*
 * Logger.cc
 *
 *  Created on: 25.11.2016
 *      Author: Max
 */

#include "Logger.h"
#include "Settings.h"
#include "boost/date_time/posix_time/posix_time.hpp"
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time.hpp>

boost::mutex Logger::m_mutex;
boost::thread* Logger::m_ownThread(nullptr);
bool Logger::m_init(false);
bool Logger::m_needToWrite(false);
std::string Logger::m_text("");
std::string Logger::m_fileName("");
double Logger::m_timeToSleep(2.);
std::map<std::string, std::string> Logger::m_specialLines;
std::string Logger::m_actualDirectory = "./"; // default

Logger::Logger() {
}

Logger::~Logger() {
}

void Logger::start(){
	m_init = Settings::getDirectBoolValue("Logger.useLogger");

	m_actualDirectory = "./"; // default
	if(m_init){
		Settings::getValue("Logger.fileName", m_fileName);

		boost::gregorian::date dayte(boost::gregorian::day_clock::local_day());
		boost::posix_time::ptime midnight(dayte);
		boost::posix_time::ptime
		now(boost::posix_time::microsec_clock::local_time());
		boost::posix_time::time_duration td = now - midnight;
		std::stringstream clockTime;
		if(td.fractional_seconds() > 0){
			const char cFracSec = number2String(td.fractional_seconds())[0];
			clockTime << td.hours() << ":" << td.minutes() << ":" << td.seconds() << "." << cFracSec;
		}else{
			clockTime << td.hours() << ":" << td.minutes() << ":" << td.seconds() << "." << 0;
		}
		for(auto&& folder : {number2String(dayte.year()),
			number2String(dayte.month().as_number()),
			number2String(dayte.day()),
			clockTime.str()} ){
			m_actualDirectory += folder + "/";
			if(!boost::filesystem::exists(m_actualDirectory)){
				system(std::string("mkdir " + m_actualDirectory).c_str());
			}
		}
		if(boost::filesystem::exists("./latest")){
			system("rm latest");
		}
		system(std::string("ln -s " + m_actualDirectory + " latest").c_str());
		printOnScreen("The folder is: " << m_actualDirectory);
		system(std::string("cp " + Settings::getFilePath() + " " + m_actualDirectory + "usedInit.json").c_str());
		std::string mode;
		Settings::getValue("main.type", mode);
		m_text = "Online Random Forest with IVMs, mode: " + mode + "\n"; // Standart Information
		m_text += "Date: " + boost::posix_time::to_simple_string(boost::posix_time::second_clock::local_time()) + "\n";
		m_ownThread = new boost::thread(&Logger::run);
	}
}

void Logger::forcedWrite(){
	if(m_init){
		m_mutex.lock();
		write();
		m_mutex.unlock();
	}
}

void Logger::write(){
	// not locked!
	std::fstream file;
	file.open(getActDirectory() + m_fileName, std::fstream::out | std::fstream::trunc);
	file.write(m_text.c_str(), m_text.length());
	for(std::map<std::string, std::string>::const_iterator it = m_specialLines.cbegin(); it != m_specialLines.cend(); ++it){
		if(!(it->first == "Error" || it->first == "Warning")){
			file << it->first << "\n";
			file.write(it->second.c_str(), it->second.length());
		}
	}
	for(auto&& name : {"Warning", "Error"}){
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
