/*
 * Logger.h
 *
 *  Created on: 25.11.2016
 *      Author: Max
 */

#ifndef BASE_LOGGER_H_
#define BASE_LOGGER_H_

#include <map>
#include <atomic>
#include "BaseType.h"
#include "Types.h"
#include "Singleton.h"
#include "Thread.h"

class Logger : public Singleton<Logger> {

	friend class Singleton<Logger>;

public:

	// checks the values in the settings file
	void start();

	void addNormalLineToFile(const std::string& line);

	void addSpecialLineToFile(const std::string& line, const std::string& identifier);

	void forcedWrite();

	std::string nameOfLogFile(){ return m_fileName; };

	bool isUsed(){ return m_init; }

	std::string getActDirectory(){ return m_actualDirectory; };

	void stopLogger(){ m_running = false;
		if(m_ownThread){
			m_ownThread->join();
			m_ownThread.reset(nullptr);
		}
	}

private:

	void run();

	void write();

	std::string m_actualDirectory;

	Mutex m_mutex;

	bool m_init;

	bool m_needToWrite;

	std::string m_text;

	std::string m_fileName;

	const Real m_timeToSleep;

	using InnerSpecialLines = std::vector< std::pair<std::string, unsigned int> >;
	std::map<std::string, InnerSpecialLines> m_specialLines;

	UniquePtr<Thread> m_ownThread;

	std::atomic<bool> m_writeByForceWrite;

	UniquePtr<std::ofstream> m_file;

	bool m_running;

	Logger();
	~Logger() = default;
};

#endif /* BASE_LOGGER_H_ */
