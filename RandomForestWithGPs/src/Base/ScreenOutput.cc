/*
 * ScreenOutput.cc
 *
 *  Created on: 24.11.2016
 *      Author: Max
 */

#include "ScreenOutput.h"
#include "Settings.h"
#include "../Utility/Util.h"

std::list<std::string> ScreenOutput::m_lines;
ThreadMaster::PackageList* ScreenOutput::m_runningThreads(nullptr);
boost::thread* ScreenOutput::m_mainThread(nullptr);
double ScreenOutput::m_timeToSleep(0.1);
boost::mutex ScreenOutput::m_lineMutex;
std::list<std::pair<std::string, StopWatch> > ScreenOutput::m_errorLines;
std::string ScreenOutput::m_progressLine;

ScreenOutput::ScreenOutput() {
	// TODO Auto-generated constructor stub

}

ScreenOutput::~ScreenOutput() {
	// TODO Auto-generated destructor stub
}

void ScreenOutput::quitForScreenMode(){
	endwin();
}

void ScreenOutput::start(){
	m_runningThreads = &ThreadMaster::m_runningList;
	initscr();
	start_color();
	init_pair(1, COLOR_GREEN, COLOR_BLACK);
	init_pair(2, COLOR_RED, COLOR_BLACK);
	init_pair(6, COLOR_CYAN, COLOR_BLACK);
	init_color(COLOR_RED, 0, 100, 100);
	init_color(COLOR_GREEN, 100, 1000, 100);
	attron(COLOR_PAIR(1));
	atexit(ScreenOutput::quitForScreenMode);
//	curs_set(0);
	m_mainThread = new boost::thread(&ScreenOutput::run);
}

void ScreenOutput::run(){
	std::string mode;
	Settings::getValue("main.type", mode);
	const std::string firstLine = "Online Random Forest with IVMs, mode: " + mode;
	int deepestActLineForProgress = 0;
	while(true){
		int actLine = 3;
		clear();
		attron(COLOR_PAIR(1));
		mvprintw(actLine++,5, firstLine.c_str());
		const std::string amountOfThreadsString = "Amount of running Threads: ";
		mvprintw(actLine,5, amountOfThreadsString.c_str());
		attron(A_BOLD);
		attron(COLOR_PAIR(6));
		mvprintw(actLine++,amountOfThreadsString.length() + 5, number2String(m_runningThreads->size()).c_str());
		attroff(A_BOLD);
		attron(COLOR_PAIR(1));
		actLine++;

		ThreadMaster::m_mutex.lock();
		int rowCounter = 0;
		const int col = COLS - 6; // 3 on both sides
		const int startOfRight = COLS / 2 + 2;
		const int amountOfLinesPerThread =  m_runningThreads->size() > 1 ?  HEIGHT_OF_THREAD_INFO / ceil(m_runningThreads->size() / 2.0) : HEIGHT_OF_THREAD_INFO;
		for(ThreadMaster::PackageList::const_iterator it = m_runningThreads->begin(); it != m_runningThreads->end(); ++it, ++rowCounter){
			(*it)->m_lineMutex.lock();
			const bool isLeft = rowCounter % 2 == 0;
			const int colWidth = col / 2 - 2;
			const int row = rowCounter / 2; // for 0 and 1 the first and so on
			if((*it)->m_standartInfo.length() < colWidth && (*it)->m_standartInfo.length() > 0){
				mvprintw(row * amountOfLinesPerThread + actLine, isLeft ? 3 : startOfRight, (*it)->m_standartInfo.c_str()); //  isLeft ? 3 : startOfRight
				int counter = 1;
				std::list<std::string>::const_iterator itLine = (*it)->m_lines.begin();
				for(int i = 0; i <= (int) (*it)->m_lines.size() - amountOfLinesPerThread + 1; ++i){
					++itLine; // jump over elements in the list which are no needed at the moment
				}
				for(; itLine != (*it)->m_lines.end(); ++itLine, ++counter){
					if(itLine->length() < colWidth){
						mvprintw(row * amountOfLinesPerThread + counter + actLine, isLeft ? 3 : startOfRight, itLine->c_str());
					}else{
						std::string line = *itLine;
						while(line.length() > colWidth){
							mvprintw(row * amountOfLinesPerThread + counter + actLine, isLeft ? 3 : startOfRight, line.substr(0, colWidth).c_str());
							line = line.substr(colWidth, line.length() - colWidth);
							++counter;
						}
						mvprintw(row * amountOfLinesPerThread + counter + actLine, isLeft ? 3 : startOfRight, line.substr(0, colWidth).c_str());
						++counter;
					}
				}
				const int diff = (*it)->m_lines.size() - HEIGHT_OF_THREAD_INFO;
				for(int i = 0; i < diff; ++i){
					(*it)->m_lines.pop_front(); // to reduce it to HEIGHT OF THREAD INFO is the maximal which can be displayed so the rest can be erased
				}
			}
			(*it)->m_lineMutex.unlock();
		}
		ThreadMaster::m_mutex.unlock();
		m_lineMutex.lock();

		actLine += HEIGHT_OF_THREAD_INFO;

		int maxSize = 0;
		for(std::list<std::string>::const_iterator it = m_lines.begin(); it != m_lines.end(); ++it){
			if(it->length() > maxSize){
				maxSize = it->length();
			}
		}
		for(std::list<std::string>::const_iterator it = m_lines.begin(); it != m_lines.end(); ++it){
			mvprintw(actLine++, 6, it->c_str());
		}
		if(m_errorLines.size() > 0){
			attron(COLOR_PAIR(2));
			mvprintw(actLine++, 3, "-------------------------------------------------------------------------------------------");
			for(std::list<std::pair<std::string, StopWatch> >::const_iterator it = m_errorLines.begin(); it != m_errorLines.end(); ++it){
				if(it->second.elapsedSeconds() > 15.){
					std::list<std::pair<std::string, StopWatch> >::const_iterator copyIt = it;
					--it;
					m_errorLines.erase(copyIt);
					continue;
				}else{
					mvprintw(actLine++, 6, it->first.c_str());
				}
			}
			mvprintw(actLine++, 3, "-------------------------------------------------------------------------------------------");
			attron(COLOR_PAIR(1));
		}
		if(actLine + 2> deepestActLineForProgress){
			deepestActLineForProgress = actLine + 2;
		}
		mvprintw(deepestActLineForProgress, 4, m_progressLine.c_str());
		m_lineMutex.unlock();
		refresh();
		usleep(m_timeToSleep * 1e6);
	}
}

void ScreenOutput::print(const std::string& line){
	m_lineMutex.lock();
	if(m_lines.size() >= HEIGHT_OF_GENERAL_INFO){
		m_lines.pop_front();
	}
	m_lines.push_back(line);
	m_lineMutex.unlock();
}

void ScreenOutput::printErrorLine(const std::string& line){
	m_lineMutex.lock();
	// the stopwatch takes the time to save how long the error should be displayed
	m_errorLines.push_back(std::pair<std::string, StopWatch>(line, StopWatch()));
	m_lineMutex.unlock();
}


void ScreenOutput::printInProgressLine(const std::string& line){
	m_lineMutex.lock();
	m_progressLine = line;
	m_lineMutex.unlock();
}
