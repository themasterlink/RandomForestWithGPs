/*
 * ScreenOutput.h
 *
 *  Created on: 24.11.2016
 *      Author: Max
 */

#ifndef BASE_SCREENOUTPUT_H_
#define BASE_SCREENOUTPUT_H_

#include <curses.h>
#include "ThreadMaster.h"
#include "../Utility/StopWatch.h"

#define HEIGHT_OF_GENERAL_INFO 100
#define MAX_HEIGHT_OF_ERROR 10

class ScreenOutput {
public:
	// must be called to get the needed information
	static void start();

	// prints a basic line to the screen
	static void print(const std::string& line);

	static void printErrorLine(const std::string& line);

	static void printInProgressLine(const std::string& line);

private:

	// is interally called by the curses library
	static void quitForScreenMode();

	static void run();

	static boost::mutex m_lineMutex;

	static std::list<std::string> m_lines;

	static std::list<std::pair<std::string, StopWatch> > m_errorLines;

	static ThreadMaster::PackageList* m_runningThreads;

	static boost::thread* m_mainThread;

	static double m_timeToSleep;

	static std::string m_progressLine;

	ScreenOutput();
	virtual ~ScreenOutput();
};

// do while is used here to avoid that str is used multiple times in the same context
#define printOnScreen(message) \
		do{std::stringstream str; str << message; ScreenOutput::print(str.str()); }while(false) \

#endif /* BASE_SCREENOUTPUT_H_ */
