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

class Logger {
public:

	// checks the values in the settings file
	static void start();

	static void addNormalLineToFile(const std::string& line);

	static void addSpecialLineToFile(const std::string& line, const std::string& identifier);

	static void setTimeBetweenWritingIntervals(const double timeToSleep){ m_timeToSleep = timeToSleep; };

	static void forcedWrite();

	static std::string nameOfLogFile(){ return m_filePath; };

	static bool isUsed(){ return m_init;}

private:

	static void write();

	static void run();

	static boost::mutex m_mutex;

	static boost::thread* m_ownThread;

	static bool m_init;

	static bool m_needToWrite;

	static std::string m_text;

	static std::string m_filePath;

	static double m_timeToSleep;

	static std::map<std::string, std::string> m_specialLines;

	Logger();
	virtual ~Logger();
};

#endif /* BASE_LOGGER_H_ */
