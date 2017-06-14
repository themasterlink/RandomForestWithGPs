/*
 * Logger.cc
 *
 *  Created on: 25.11.2016
 *      Author: Max
 */

#include "Logger.h"
#include "Settings.h"
#include <boost/date_time.hpp>

boost::mutex Logger::m_mutex;
bool Logger::m_init(false);
bool Logger::m_needToWrite(false);
std::string Logger::m_text("");
std::string Logger::m_fileName("");
Real Logger::m_timeToSleep((Real) 2.);
std::map<std::string, std::string> Logger::m_specialLines;
std::string Logger::m_actualDirectory = "./"; // default
boost::thread* Logger::m_ownThread(nullptr);

namespace bfs = boost::filesystem;

Logger::Logger() {
}

Logger::~Logger() {
}

void Logger::start(){
	m_init = Settings::getDirectBoolValue("Logger.useLogger");

	auto createFolder = [](const std::string& dir){
		if(!bfs::exists(dir)){
			bfs::create_directory(dir);
		}
	};
#ifdef BUILD_SYSTEM_LINUX
	// "Linux"
	m_actualDirectory = "/home_local/denn_ma"; // default
	createFolder(m_actualDirectory);
	m_actualDirectory += "/log";
	createFolder(m_actualDirectory);
	m_actualDirectory += "/";
#else // Something else
	m_actualDirectory = "./"
#endif
	if(m_init){
		Settings::getValue("Logger.fileName", m_fileName);
		boost::gregorian::date dayte(boost::gregorian::day_clock::local_day());
		for(auto&& folder : {StringHelper::number2String(dayte.year()),
			StringHelper::number2String(dayte.month().as_number()),
			StringHelper::number2String(dayte.day()),
			StringHelper::getActualTimeOfDayAsString()} ){
			m_actualDirectory += folder + "/";
			createFolder(m_actualDirectory);
		}
//		if(bfs::exists("./latest")){
//			system("rm latest");
//		}
//		system(std::string("ln -s " + m_actualDirectory + " latest").c_str());
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
	std::string out;
//	file.write(m_text.c_str(), m_text.length());
	out += "This file was written at: " + StringHelper::getActualTimeOfDayAsString() + "\n";
	for(const auto& line : m_specialLines){
		if(!(line.first == "Error" || line.first == "Warning")){
			out += line.first + "\n" + line.second;
//			file << line.first << "\n";
//			file.write(line.second.c_str(), line.second.length());
		}
	}
	for(auto&& name : {"Warning", "Error"}){
		auto itOther = m_specialLines.find(name);
		if(itOther != m_specialLines.end()){
			out += itOther->first + "\n" + itOther->second;
//			file << itOther->first << "\n";
//			file.write(itOther->second.c_str(), itOther->second.length());
		}
	}
	const std::string oldFileName = m_fileName;
	m_fileName = "log" + StringHelper::getActualTimeOfDayAsString() + ".txt";
	const auto fileLoc = getActDirectory() + m_fileName;
	const auto fileLocOld = getActDirectory() + oldFileName;
//	if(bfs::exists(fileLoc)){
//		std::rename(fileLoc.c_str(), fileLocOld.c_str());
//	}
	std::fstream file;
	//prepare f to throw if failbit gets set
	const auto exceptionMask = file.exceptions() | std::ios::failbit;
	file.exceptions(exceptionMask);
	try{
		file.open(fileLoc, std::fstream::out | std::fstream::trunc);
	}catch(std::fstream::failure& e){
		m_mutex.unlock();
		printError("The log file opening failed with: " << e.what());
		m_mutex.lock();
	}
	if(file.is_open()){
		file.write(out.c_str(), out.length());
		file.close();
		std::string latestFile = getActDirectory() + "log.txt";
		if(bfs::exists(latestFile)){
			bfs::remove(latestFile);
		}
		bfs::create_symlink(fileLoc, latestFile);
		system(std::string("zip " + getActDirectory() + "log.zip " + fileLoc + " &>/dev/null 2>&1").c_str());
		if(bfs::exists(fileLocOld)){
			bfs::remove(fileLocOld);
		}
	}else{
		m_mutex.unlock();
		printError("The log file could not be opened!");
		m_mutex.lock();
	}
	m_needToWrite = false;
}

void Logger::run(){
	while(m_init){
		m_mutex.lock();
		if(m_needToWrite){ // write only if something changed
			write();
		}
		m_mutex.unlock();
		sleepFor(m_timeToSleep);
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
		auto it = m_specialLines.find(identifier);
		if(it != m_specialLines.end()){
			it->second += ("\t" + line + "\n");
		}else{
			const std::string input = "\t" + line + "\n";
			m_specialLines.emplace(identifier, input);
		}
		m_needToWrite = true;
		m_mutex.unlock();
	}
}
