/*
 * ScreenOutput.h
 *
 *  Created on: 24.11.2016
 *      Author: Max
 */

#ifndef BASE_SCREENOUTPUT_H_
#define BASE_SCREENOUTPUT_H_

//#define NO_OUTPUT

#include <curses.h>
#include <panel.h>
#undef OK
#include "ThreadMaster.h"
#include "../Utility/StopWatch.h"
#include "Types.h"
#include "Singleton.h"
#include "Thread.h"

#define MAX_HEIGHT 120

class ScreenOutput : public Singleton<ScreenOutput> {

	friend class Singleton<ScreenOutput>;

public:
	// must be called to get the needed information
	void start();

	// prints a basic line to the screen
	void print(const std::string& line);

	void printErrorLine(const std::string& line);

	void printInProgressLine(const std::string& line);

	// is interally called by the curses library, must be static therefore
	static void quitForScreenMode();

	void blockUntilStopped(){ // is automatically stopped when ThreadMaster is stopped
		if(m_mainThread){
			m_mainThread->join();
			m_mainThread.reset(nullptr);
		}
	}

	// just function to make a connection to Logger
	void printToLog(const std::string& message);

	// just function to make a connection to Logger
	void printToLogSpecial(const std::string& message, const std::string& special);

private:

	void run();

	void fillWindow(WINDOW* win, const std::list<std::string>& lines, const int width, const int height,
					const bool hasHeadLine, const int color = 1);

	void
	drawWindow(WINDOW** window, PANEL** panel, int givenHeight, int givenWidth, int x, int y, const bool hasHeadLine,
			   const int color = 1);

	void updateRunningPackage(ThreadMaster::PackageListConstIterator& it, const int rowCounter, const int row,
							  const bool isLeft, const int colWidth,
							  const int amountOfLinesPerThread, int& actLine, int startOfRight, std::vector<WINDOW*>& windows, std::vector<PANEL*>& panels);

	Mutex m_lineMutex;

	std::list<std::string> m_lines;

	std::list<std::string> m_errorLines;

	std::list<int> m_errorCounters;

	ThreadMaster::PackageList* m_runningThreads;

	UniquePtr<Thread> m_mainThread;

	Real m_timeToSleep;

	std::string m_progressLine;

	ScreenOutput();
	~ScreenOutput() = default;

};

#ifdef USE_SCREEN_OUPUT
// do while is used here to avoid that str is used multiple times in the same context
#define printOnScreen(message) \
        do{std::stringstream str; str << message; ScreenOutput::instance().print(str.str()); }while(false) \

#define printInPackageOnScreen(package, message) \
		do{std::stringstream str; str << message; (package)->printLineToScreenForThisThread(str.str()); }while(false) \

#else // USE_SCREEN_OUPUT
#define printOnScreen(message) \
	do{	std::stringstream str100; str100 << message; std::cout << str100.str() << std::endl; \
		ScreenOutput::instance().printToLog(str100.str()); }while(false) \

#define printInPackageOnScreen(package, message) \
	do{std::stringstream str100; str100 << message; \
		std::cout << (package)->getStandartInformation() << "\n\t" << str100.str() << std::endl; \
		ScreenOutput::instance().printToLogSpecial(str100.str(), (package)->getStandartInformation()); }while(false) \

#endif // USE_SCREEN_OUPUT

#endif /* BASE_SCREENOUTPUT_H_ */
