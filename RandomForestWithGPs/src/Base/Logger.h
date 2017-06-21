/*
 * Logger.h
 *
 *  Created on: 25.11.2016
 *      Author: Max
 */

#ifndef BASE_LOGGER_H_
#define BASE_LOGGER_H_

#include <boost/thread.hpp>
#include <map>
#include "BaseType.h"

class Logger {
public:

	// checks the values in the settings file
	static void start();

	static void addNormalLineToFile(const std::string& line);

	static void addSpecialLineToFile(const std::string& line, const std::string& identifier);

	static void forcedWrite();

	static std::string nameOfLogFile(){ return m_fileName; };

	static bool isUsed(){ return m_init;}

	static std::string getActDirectory(){ return m_actualDirectory; };

private:

	static void write();

	static void run();

	static std::string m_actualDirectory;

	static boost::mutex m_mutex;

	static bool m_init;

	static bool m_needToWrite;

	static std::string m_text;

	static std::string m_fileName;

	static const Real m_timeToSleep;

	static std::map<std::string, std::string> m_specialLines;

	Logger();
	virtual ~Logger();

	static boost::thread *m_ownThread;

	static std::atomic<bool> m_writeByForceWrite;

	static std::unique_ptr<std::ofstream> m_file;
};

#endif /* BASE_LOGGER_H_ */
