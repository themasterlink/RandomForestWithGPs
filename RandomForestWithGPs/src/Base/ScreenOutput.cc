/*
 * ScreenOutput.cc
 *
 *  Created on: 24.11.2016
 *      Author: Max
 */

#include "ScreenOutput.h"
#include "Settings.h"
#include "Logger.h"
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
	clear();
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
	while(true){
		clear();
		const int progressBar = LINES - 3;
		const int heightOfThreadInfo = LINES * 0.7;
		const int restOfLines = LINES - 7 - heightOfThreadInfo;
		if(heightOfThreadInfo < 25 && restOfLines > 3 && COLS < 80){
			mvprintw(1,0, firstLine.c_str());
			mvprintw(2,0, "There is not enough room to fill the rest with information!");
			refresh();
			usleep(m_timeToSleep * 1e6);
			continue;
		}

		int actLine = 1;
		mvprintw(actLine++,5, firstLine.c_str());
		const std::string amountOfThreadsString = "Amount of running Threads: ";
		ThreadMaster::m_mutex.lock();
		mvprintw(actLine,5, amountOfThreadsString.c_str());
		attron(A_BOLD);
		attron(COLOR_PAIR(6));
		mvprintw(actLine++,amountOfThreadsString.length() + 5, number2String(m_runningThreads->size()).c_str());
		attroff(A_BOLD);
		attron(COLOR_PAIR(1));
		actLine++;

		int rowCounter = 0;
		const int col = COLS - 6; // 3 on both sides
		const int startOfRight = COLS / 2 + 2;
		const int amountOfLinesPerThread =  m_runningThreads->size() > 1 ?  heightOfThreadInfo / ceil(m_runningThreads->size() / 2.0) : heightOfThreadInfo;
		for(ThreadMaster::PackageList::const_iterator it = m_runningThreads->begin(); it != m_runningThreads->end(); ++it, ++rowCounter){
			(*it)->m_lineMutex.lock();
			const bool isLeft = rowCounter % 2 == 0;
			const int colWidth = col / 2 - 6; // -2 on both sides
			const int row = rowCounter / 2; // for 0 and 1 the first and so on
			std::string drawLine = "";

			for(unsigned int i = 0; i < colWidth + (isLeft ? 4 : 3); ++i){
				drawLine += "-";
			}
			mvprintw(row * amountOfLinesPerThread + actLine - 1, isLeft ? 2 : startOfRight - 1, drawLine.c_str()); //  isLeft ? 3 : startOfRight
			mvprintw((row + 1) * amountOfLinesPerThread + actLine - 2, isLeft ? 2 : startOfRight - 1, drawLine.c_str()); //  isLeft ? 3 : startOfRigh
			for(unsigned int i = 1; i < amountOfLinesPerThread - 1; ++i){
				mvprintw(row * amountOfLinesPerThread + actLine - 1 + i, isLeft ? 2 : startOfRight - 1, "|");
				mvprintw(row * amountOfLinesPerThread + actLine - 1 + i, isLeft ? 6 + colWidth: startOfRight + colWidth + 2, "|");
			}

			mvprintw(row * amountOfLinesPerThread + actLine - 1, isLeft ? 2 : startOfRight - 1, "+");
			mvprintw(row * amountOfLinesPerThread + actLine - 1, isLeft ? 6 + colWidth: startOfRight + colWidth + 2, "+");
			mvprintw((row + 1) * amountOfLinesPerThread + actLine - 2, isLeft ? 2 : startOfRight - 1, "+");
			mvprintw((row + 1) * amountOfLinesPerThread + actLine - 2, isLeft ? 6 + colWidth: startOfRight + colWidth + 2, "+");
			if((*it)->m_standartInfo.length() < colWidth && (*it)->m_standartInfo.length() > 0){
				attron(COLOR_PAIR(6));
				mvprintw(row * amountOfLinesPerThread + actLine, isLeft ? 4 : startOfRight + 1, (*it)->m_standartInfo.c_str()); //  isLeft ? 3 : startOfRight
				attron(COLOR_PAIR(1));
				int counter = 1;
				int amountOfNeededLines = 0;
				for(std::list<std::string>::const_iterator itLine = (*it)->m_lines.begin(); itLine != (*it)->m_lines.end(); ++itLine){
					if(itLine->length() > colWidth){
						amountOfNeededLines += ceil((itLine->length() - colWidth) / (double) (colWidth - 2)) + 1;
					}else{
						++amountOfNeededLines;
					}
				}
				if(amountOfNeededLines < amountOfLinesPerThread - 2){
					std::list<std::string>::const_iterator itLine = (*it)->m_lines.begin();
					for(int i = 0; i <= (int) (*it)->m_lines.size() - amountOfLinesPerThread + 2; ++i){
						++itLine; // jump over elements in the list which are no needed at the moment
					}
					for(; itLine != (*it)->m_lines.end(); ++itLine, ++counter){
						if(itLine->length() < colWidth){
							mvprintw(row * amountOfLinesPerThread + counter + actLine, isLeft ? 4 : startOfRight + 1, itLine->c_str());
						}else{
							std::string line = *itLine;
							int diff = 0;
							while(line.length() > colWidth){
								mvprintw(row * amountOfLinesPerThread + counter + actLine, isLeft ? 4 + diff: startOfRight + 1 + diff, line.substr(0, colWidth - diff).c_str());
								++counter;
								line = line.substr(colWidth, line.length() - colWidth);
								diff = 2;
							}
							mvprintw(row * amountOfLinesPerThread + counter + actLine, isLeft ? 4 + diff: startOfRight + 1 + diff, line.substr(0, colWidth - diff).c_str());
						}
					}
				}else{
					counter += 2;
					for(std::list<std::string>::reverse_iterator itLine = (*it)->m_lines.rbegin(); itLine != (*it)->m_lines.rend(); ++itLine, ++counter){
						if(itLine->length() < colWidth){
							if(counter < amountOfLinesPerThread){ // -1 for the start line
								mvprintw((row + 1) * amountOfLinesPerThread - counter + actLine, isLeft ? 4 : startOfRight + 1, itLine->c_str());
							}
						}else{
							std::string line = *itLine;
							counter += ceil((itLine->length() - colWidth) / (double) (colWidth - 2));
							int diff = 0;
							while(line.length() > colWidth){
								if(counter < amountOfLinesPerThread){ // -1 for the start line
									mvprintw((row + 1) * amountOfLinesPerThread - counter + actLine, isLeft ? 4 + diff : startOfRight + 1 + diff, line.substr(0, colWidth - diff).c_str());
								}
								diff = 2;
								line = line.substr(colWidth, line.length() - colWidth);
								--counter;
							}
							if(counter < amountOfLinesPerThread){ // -1 for the start line
								mvprintw((row + 1) * amountOfLinesPerThread - counter + actLine, isLeft ? 4  + diff : startOfRight + 1  + diff, line.substr(0, colWidth - diff).c_str());
							}
							counter += ceil((itLine->length() - colWidth) / (double) (colWidth - 2));
						}
					}
				}
				attron(COLOR_PAIR(1));
				const int diff = (*it)->m_lines.size() - heightOfThreadInfo;
				for(int i = 0; i < diff; ++i){
					(*it)->m_lines.pop_front(); // to reduce it to HEIGHT OF THREAD INFO is the maximal which can be displayed so the rest can be erased
				}
			}
			(*it)->m_lineMutex.unlock();
		}
		ThreadMaster::m_mutex.unlock();
		m_lineMutex.lock();

		actLine += amountOfLinesPerThread * ceil(m_runningThreads->size() / 2.0) ;
		int maxSize = 0;
		for(std::list<std::string>::const_iterator it = m_lines.begin(); it != m_lines.end(); ++it){
			if(it->length() > maxSize){
				maxSize = it->length();
			}
		}
		int iLineCounter = std::min((int) LINES - actLine - 7 - (int) m_errorLines.size(), (int) m_lines.size());
		for(std::list<std::string>::reverse_iterator it = m_lines.rbegin(); it != m_lines.rend() && iLineCounter >= 0; ++it){
			mvprintw(actLine + iLineCounter, 6, it->c_str());
			--iLineCounter;
		}
		actLine += std::min((int) LINES - actLine - 7 - (int) m_errorLines.size(), (int) m_lines.size()) + 1;

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
		mvprintw(progressBar, 4, m_progressLine.c_str());
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
	Logger::addToFile(line);
}

void ScreenOutput::printErrorLine(const std::string& line){
	m_lineMutex.lock();
	if(m_errorLines.size() >= MAX_HEIGHT_OF_ERROR){
		m_errorLines.pop_front();
	}
	// the stopwatch takes the time to save how long the error should be displayed
	m_errorLines.push_back(std::pair<std::string, StopWatch>(line, StopWatch()));
	m_lineMutex.unlock();
	Logger::addToFile(line);
}


void ScreenOutput::printInProgressLine(const std::string& line){
	m_lineMutex.lock();
	m_progressLine = line;
	m_lineMutex.unlock();
}
