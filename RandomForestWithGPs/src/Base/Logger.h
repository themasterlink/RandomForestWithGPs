/*
 * Logger.h
 *
 *  Created on: 25.11.2016
 *      Author: Max
 */

#ifndef BASE_LOGGER_H_
#define BASE_LOGGER_H_

#include <boost/thread.hpp>

class Logger {
public:

	// checks the values in the settings file
	static void init();

	static void addToFile(const std::string& line);

	static void setTimeBetweenWritingIntervals(const double timeToSleep){ m_timeToSleep = timeToSleep; };

private:

	static void run();

	static boost::mutex m_mutex;

	static boost::thread* m_ownThread;

	static bool m_init;

	static bool m_needToWrite;

	static std::string m_text;

	static std::string m_filePath;

	static double m_timeToSleep;

	Logger();
	virtual ~Logger();
};

#endif /* BASE_LOGGER_H_ */
