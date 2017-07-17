/*
 * Logger.cc
 *
 *  Created on: 25.11.2016
 *      Author: Max
 */

#include "Logger.h"
#include "Settings.h"
#include <boost/date_time.hpp>
#include "../Tests/TestManager.h"

namespace bfs = boost::filesystem;

Logger::Logger():
		m_init(false),
		m_needToWrite(false),
		m_text(""),
		m_fileName(""),
		m_timeToSleep(1.2_r/*5 * 60.0*/),
		m_actualDirectory("./"), // default
		m_ownThread(nullptr),
		m_writeByForceWrite(true),
		m_file(nullptr){
}

void Logger::start(){
	systemCall("ulimit -Sf 2>&1 >0"); 	// sets the soft limit for the file handles to unlimited,
										// so no problems for the logging arise
	m_init = Settings::instance().getDirectBoolValue("Logger.useLogger");
	if(m_init){
		auto createFolder = [](const std::string& dir){
			if(!bfs::exists(dir)){
				bfs::create_directory(dir);
			}
		};
#if(BUILD_SYSTEM_CMAKE == 1)
		// "Linux"
		m_actualDirectory = "/home_local/denn_ma"; // default
		createFolder(m_actualDirectory);
		m_actualDirectory += "/log";
		createFolder(m_actualDirectory);
		m_actualDirectory += "/";
#else // Something else
		m_actualDirectory = "./";
#endif
		Settings::instance().getValue("Logger.fileName", m_fileName);
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
//		bfs::copy_file(Settings::instance().getFilePath(), m_actualDirectory + "usedInit.json");
		system(std::string(
				"cp " + Settings::instance().getFilePath() + " " + m_actualDirectory + "usedInit.json").c_str());
		system(std::string(
				"cp " + TestManager::instance().getFilePath() + " " + m_actualDirectory + "testSettings.txt").c_str());
		std::string mode;
		Settings::instance().getValue("main.type", mode);
		m_text = "Online Random Forest with IVMs, mode: " + mode + "\n"; // Standart Information
		m_text += "Date: " + boost::posix_time::to_simple_string(boost::posix_time::second_clock::local_time()) + "\n";
		m_ownThread = makeThread(&Logger::run, &Logger::instance());
	}
}

void Logger::forcedWrite(){
	if(m_init && m_writeByForceWrite){
		lockStatementWith(write(), m_mutex);
	}
}

void Logger::write(){
	// not locked!
	std::string out;
//	m_file.write(m_text.c_str(), m_text.length());
	out += "This m_file was written at: " + StringHelper::getActualTimeOfDayAsString() + "\n";
	out += m_text;
	for(const auto& ele : m_specialLines){
		if(!(ele.first == "Error" || ele.first == "Warning")){
			out += ele.first + "\n";
			for(const auto& lines : ele.second){
				if(lines.second < 2){
					out += "\t" + lines.first + "\n";
				}else{
					out += "\t" + lines.first + ", occured " + StringHelper::number2String(lines.second) + " times\n";
				}
			}
		}
	}
	for(auto&& name : {"Warning", "Error"}){
		auto itOther = m_specialLines.find(name);
		if(itOther != m_specialLines.end()){
			out += itOther->first + "\n";
			for(const auto& lines : itOther->second){
				if(lines.second < 2){
					out += "\t" + lines.first + "\n";
				}else{
					out += "\t" + lines.first + ", occured " + StringHelper::number2String(lines.second) + " times\n";
				}
			}
		}
	}
	const std::string oldFileName = m_fileName;
	m_fileName = "log.txt"; // " + StringHelper::getActualTimeOfDayAsString() + "
	const auto fileLoc = getActDirectory() + m_fileName;
	const auto fileLocOld = getActDirectory() + oldFileName;
//	if(bfs::exists(fileLoc)){
//		std::rename(fileLoc.c_str(), fileLocOld.c_str());
//	}
	m_file = std::make_unique<std::ofstream>();
	//prepare f to throw if failbit gets set
	const auto exceptionMask = m_file->exceptions() | std::ios::failbit;
	m_file->exceptions(exceptionMask);
	try{
		m_file->open(fileLoc, std::ofstream::out | std::ofstream::trunc);
	}catch(std::ofstream::failure& e){
		m_mutex.unlock();
		printErrorAndQuit("The log m_file opening failed with: " << e.what() << ", errno: " << strerror(errno));
		m_mutex.lock();
	}
	if(m_file->is_open()){
		m_file->write(out.c_str(), out.length());
		m_file.reset(nullptr); // destroys the ofstream object -> closes the m_file handle?
		/*std::string latestFile = getActDirectory() + "log.txt";
		if(bfs::exists(latestFile)){
			bfs::remove(latestFile);
		}
		try{
			bfs::create_symlink(fileLoc, latestFile);
		}catch(bfs::filesystem_error& e){
			m_mutex.unlock();
			printError("The create symlink failed: " << e.what());
			m_mutex.lock();
		}
//		system(std::string("zip " + getActDirectory() + "log.zip " + fileLoc + " &>/dev/null 2>&1").c_str());
		if(bfs::exists(fileLocOld)){
			bfs::remove(fileLocOld);
		}*/
	}else{
		m_mutex.unlock();
		m_writeByForceWrite = false; // to avoid that the quit of the program calls the force write method
		printErrorAndQuit("The log m_file is closed, errno: " << strerror(errno));
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
			auto& ref = it->second;
			bool found = false;
			for(auto& ele : ref){
				if(ele.first == line){
					++ele.second;
					found = true;
					break;
				}
			}
			if(!found){
				// add line
				ref.emplace_back(line, 1);
			}
		}else{
			InnerSpecialLines lines;
			lines.emplace_back(line, 1);
			m_specialLines.emplace(identifier, lines);
		}
		m_needToWrite = true;
		m_mutex.unlock();
	}
}
