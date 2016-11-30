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
double ScreenOutput::m_timeToSleep(0.06);
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
	std::vector<WINDOW*> windows(8, nullptr);
	std::vector<PANEL*> panels(8, nullptr);
	std::string mode;
	Settings::getValue("main.type", mode);
	const std::string firstLine = "Online Random Forest with IVMs, mode: " + mode;
	noecho();
	cbreak();	/* Line buffering disabled. pass on everything */
	nodelay(stdscr, true);
	keypad(stdscr, true);
	int lineOffset = 0;
	int modeNr = -1;
	StopWatch sw;
	while(true){
		mvprintw(1,5, firstLine.c_str());
//		clear();
		std::string completeWhite = "";
		for(unsigned int i = 0; i < COLS; ++i){
			completeWhite += " ";
		}
		const int progressBar = LINES - 3;
		const int heightOfThreadInfo = LINES * 0.7;
		const int restOfLines = LINES - 7 - heightOfThreadInfo;
		if(heightOfThreadInfo < 25 && restOfLines > 3 && COLS < 80){
			clear();
			mvprintw(1,0, firstLine.c_str());
			mvprintw(2,0, "There is not enough room to fill the rest with information!");
			refresh();
			usleep(m_timeToSleep * 1e6);
			continue;
		}
		int actLine = 2;
		mvprintw(actLine,0, completeWhite.c_str());
		int x,y;
		getyx(stdscr, y, x);
		const std::string amountOfThreadsString = "Amount of running Threads: ";
		ThreadMaster::m_mutex.lock();
		mvprintw(actLine, 5, amountOfThreadsString.c_str());
		attron(A_BOLD);
		attron(COLOR_PAIR(6));
		const std::string runningThreadAsString = number2String(m_runningThreads->size()) + ", " + number2String(ThreadMaster::m_waitingList.size());
		mvprintw(actLine,amountOfThreadsString.length() + 5, runningThreadAsString.c_str());
		attroff(A_BOLD);
		attron(COLOR_PAIR(1));
		if(m_runningThreads->size() == 0 || modeNr + 1 > m_runningThreads->size()){
			modeNr = -1;
		}
		if(modeNr != -1){
			mvprintw(actLine,amountOfThreadsString.length() + 5 + runningThreadAsString.length(), (", mode: " + number2String(modeNr + 1)).c_str());
		}
		actLine += 2;
		int rowCounter = 0;
		const int col = COLS - 6; // 3 on both sides
		const int startOfRight = COLS / 2 + 2;
		const int amountOfLinesPerThread =  m_runningThreads->size() > 1 ?  heightOfThreadInfo / ceil(m_runningThreads->size() / 2.0) : heightOfThreadInfo;
		if(modeNr == -1){
			for(ThreadMaster::PackageList::const_iterator it = m_runningThreads->begin(); it != m_runningThreads->end(); ++it, ++rowCounter){
				const bool isLeft = rowCounter % 2 == 0;
				const int row = rowCounter / 2; // for 0 and 1 the first and so on
				updateRunningPackage(it, rowCounter, row, isLeft, m_runningThreads->size() > 1 ? col / 2 : col, amountOfLinesPerThread, actLine, startOfRight, windows, panels);
			}
			for(unsigned int i = m_runningThreads->size(); i < 8; ++i){
				if(panels[i] != nullptr){
					hide_panel(panels[i]);
				}
			}
		}else{
			for(ThreadMaster::PackageList::const_iterator it = m_runningThreads->begin(); it != m_runningThreads->end(); ++it, ++rowCounter){
				if(modeNr == rowCounter){
					updateRunningPackage(it, rowCounter, 0, true, col, heightOfThreadInfo, actLine, startOfRight, windows, panels);
				}
			}
			for(unsigned int i = 0; i < 8; ++i){
				if(panels[i] != nullptr && i != modeNr){
					hide_panel(panels[i]);
				}
			}
		}
		for(ThreadMaster::PackageList::const_iterator it = m_runningThreads->begin(); it != m_runningThreads->end(); ++it, ++rowCounter){
			(*it)->m_lineMutex.lock();
			const int diff = (*it)->m_lines.size() - heightOfThreadInfo;
			for(int i = 0; i < diff; ++i){
				(*it)->m_lines.pop_front(); // to reduce it to HEIGHT OF THREAD INFO is the maximal which can be displayed so the rest can be erased
			}
			(*it)->m_lineMutex.unlock();
		}
//		doupdate();
		ThreadMaster::m_mutex.unlock();
		m_lineMutex.lock();
		// remove the general information from the last call
		for(unsigned int i = actLine; i < progressBar; ++i){
			mvprintw(i, 0, completeWhite.c_str());
		}
		actLine += amountOfLinesPerThread * ceil(m_runningThreads->size() / 2.0) ;
		int maxSize = 0;
		for(std::list<std::string>::const_iterator it = m_lines.begin(); it != m_lines.end(); ++it){
			if(it->length() > maxSize){
				maxSize = it->length();
			}
		}
		int iLineCounter = std::min((int) LINES - actLine - 7 - (int) m_errorLines.size(), (int) m_lines.size());
		std::list<std::string>::reverse_iterator it = m_lines.rbegin();
		for(int i = 0; i < lineOffset; ++i){
			++it;
			if(it == m_lines.rend()){
				break;
			}
		}
		for(; it != m_lines.rend() && iLineCounter >= 0; ++it){
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
		mvprintw(progressBar,0, completeWhite.c_str());
		mvprintw(progressBar + 1,0, completeWhite.c_str());
		mvprintw(progressBar, 4, m_progressLine.c_str());
		m_lineMutex.unlock();
		update_panels();
//		doupdate();
		refresh();
		int ch = getch();
		if(ch != ERR){
			if(ch == KEY_UP){
				++lineOffset;
			}else if(ch == KEY_DOWN){
				--lineOffset;
			}else if(ch == 27){ // end of program == esc
				endwin();
				Logger::forcedWrite();
				exit(0);
			}else if(49 <= ch && ch <= 56){
				const int newNr = ch - 49;
				if(newNr == modeNr){
					modeNr = -1; // deactivate it
				}else{
					modeNr = newNr;
				}
			}
		}
		const double elapsedTime = sw.elapsedSeconds();
		if(elapsedTime < m_timeToSleep){
			usleep((m_timeToSleep - elapsedTime) * 1e6);
		}
		sw.startTime();
	}
}

void ScreenOutput::updateRunningPackage(ThreadMaster::PackageList::const_iterator& it, const int rowCounter, const int row, const bool isLeft, const int colWidth,
		const int amountOfLinesPerThread, int& actLine, int startOfRight, std::vector<WINDOW*>& windows, std::vector<PANEL*>& panels){
	(*it)->m_lineMutex.lock();
	int height, width; // in both if they are set
	WINDOW* win;
	if(windows[rowCounter] == nullptr){
		windows[rowCounter] = newwin(amountOfLinesPerThread, colWidth, row * amountOfLinesPerThread + actLine - 1, isLeft ? 2 : startOfRight - 1);
		win = windows[rowCounter];
		panels[rowCounter] = new_panel(win);
		wattron(win, COLOR_PAIR(1)); // make them green
		box(win, 0, 0);
		mvwaddch(win, 2, 0, ACS_LTEE);
		getmaxyx(win, height, width);
		mvwhline(win, 2, 1, ACS_HLINE, width - 2);
		mvwaddch(win, 2, width - 1, ACS_RTEE);
	}else{
		getmaxyx(windows[rowCounter], height, width);
		if(height != amountOfLinesPerThread || width != colWidth){
			// new draw!
			del_panel(panels[rowCounter]);
			delwin(windows[rowCounter]);
			windows[rowCounter] = newwin(amountOfLinesPerThread, colWidth, row * amountOfLinesPerThread + actLine - 1, isLeft ? 2 : startOfRight - 1);
			win = windows[rowCounter];
			panels[rowCounter] = new_panel(win);
			wattron(win, COLOR_PAIR(1)); // make them green
			box(win, 0, 0);
			mvwaddch(win, 2, 0, ACS_LTEE);
			getmaxyx(win, height, width);
			mvwhline(win, 2, 1, ACS_HLINE, width - 2);
			mvwaddch(win, 2, width - 1, ACS_RTEE);
		}else{
			win = windows[rowCounter];
		}
	}
	width -= 4; // adjust to field in the middle
	show_panel(panels[rowCounter]);
	const std::string standartLine = (*it)->m_standartInfo + ((*it)->m_additionalInformation.length() > 0 ? (", " + (*it)->m_additionalInformation) : "");
	std::string whiteSpacesForWindow;
	for(unsigned int i = 0; i < width + 1; ++i){
		whiteSpacesForWindow += " ";
	}
	mvwprintw(win, 1, 1, whiteSpacesForWindow.c_str());
	for(unsigned int i = 3; i < height - 1; ++i){
		mvwprintw(win, i, 1, whiteSpacesForWindow.c_str());
	}
	wattron(win, COLOR_PAIR(6));
	if(standartLine.length() > width){
		mvwprintw(win, 1, 1, (standartLine.substr(0, width - 3) + "...").c_str());
	}else{
		mvwprintw(win, 1, (width - standartLine.length()) / 2, standartLine.c_str());
	}
	wattron(win, COLOR_PAIR(1));
	int counter = 3;
	int amountOfNeededLines = 0;
	for(std::list<std::string>::const_iterator itLine = (*it)->m_lines.begin(); itLine != (*it)->m_lines.end(); ++itLine){
		if(itLine->length() > width){
			amountOfNeededLines += ceil((itLine->length() - width) / (double) width) + 1;
		}else{
			++amountOfNeededLines;
		}
	}
	if(amountOfNeededLines < height - 3){
		std::list<std::string>::const_iterator itLine = (*it)->m_lines.begin();
		for(int i = 0; i <= (int) (*it)->m_lines.size() - height + 1; ++i){
			++itLine; // jump over elements in the list which are no needed at the moment
		}
		for(; itLine != (*it)->m_lines.end(); ++itLine, ++counter){
			if(itLine->length() < width){
				mvwprintw(win, counter, 2, itLine->c_str());
			}else{
				std::string line = *itLine;
				int diff = 0;
				while(line.length() > width){
					mvwprintw(win, counter, 2 + diff, line.substr(0, width - diff).c_str());
					++counter;
					line = line.substr(width, line.length() - width);
					diff = 2;
				}
				mvwprintw(win, counter, 2 + diff, line.substr(0, width - diff).c_str());
			}
		}
	}else{
		counter = 2;
		for(std::list<std::string>::reverse_iterator itLine = (*it)->m_lines.rbegin(); itLine != (*it)->m_lines.rend(); ++itLine, ++counter){
			if(itLine->length() < width){
				if(counter + 2 < height){ // -1 for the start line
					mvwprintw(win, height - counter, 2, itLine->c_str());
				}
			}else{
				std::string line = *itLine;
				const int neededLen = ceil(itLine->length() / (double) width) - 1; // the last line will be printed normally
				counter += neededLen;
				int diff = 0;
				while(line.length() > width){
					if(counter + 2 < height){ // -1 for the start line
						mvwprintw(win, height - counter, 2 + diff, line.substr(0, width - diff).c_str());
					}
					diff = 2;
					line = line.substr(width, line.length() - width);
					--counter;
				}
				if(counter + 2 < height){ // -1 for the start line
					mvwprintw(win, height - counter, 2 + diff, line.substr(0, width - diff).c_str());
				}
				counter += neededLen;
			}
		}
	}
// drawing the box
//			mvprintw(row * amountOfLinesPerThread + actLine - 1, isLeft ? 2 : startOfRight - 1, drawLine.c_str()); //  isLeft ? 3 : startOfRight
//			mvprintw((row + 1) * amountOfLinesPerThread + actLine - 2, isLeft ? 2 : startOfRight - 1, drawLine.c_str()); //  isLeft ? 3 : startOfRigh
//			for(unsigned int i = 1; i < amountOfLinesPerThread - 1; ++i){
//				mvprintw(row * amountOfLinesPerThread + actLine - 1 + i, isLeft ? 2 : startOfRight - 1, "|");
//				mvprintw(row * amountOfLinesPerThread + actLine - 1 + i, isLeft ? 6 + colWidth: startOfRight + colWidth + 2, "|");
//			}
//			mvprintw(row * amountOfLinesPerThread + actLine - 1, isLeft ? 2 : startOfRight - 1, "+");
//			mvprintw(row * amountOfLinesPerThread + actLine - 1, isLeft ? 6 + colWidth: startOfRight + colWidth + 2, "+");
//			mvprintw((row + 1) * amountOfLinesPerThread + actLine - 2, isLeft ? 2 : startOfRight - 1, "+");
//			mvprintw((row + 1) * amountOfLinesPerThread + actLine - 2, isLeft ? 6 + colWidth: startOfRight + colWidth + 2, "+");
//			if((*it)->m_standartInfo.length() < colWidth && (*it)->m_standartInfo.length() > 0){
//				attron(COLOR_PAIR(6));
//				const std::string standartLine = (*it)->m_standartInfo + ((*it)->m_additionalInformation.length() > 0 ? (", " + (*it)->m_additionalInformation) : "");
//				mvprintw(row * amountOfLinesPerThread + actLine, isLeft ? 4 : startOfRight + 1, standartLine.c_str()); //  isLeft ? 3 : startOfRight
//				attron(COLOR_PAIR(1));
//				int counter = 1;
//				int amountOfNeededLines = 0;
//				for(std::list<std::string>::const_iterator itLine = (*it)->m_lines.begin(); itLine != (*it)->m_lines.end(); ++itLine){
//					if(itLine->length() > colWidth){
//						amountOfNeededLines += ceil((itLine->length() - colWidth) / (double) (colWidth - 2)) + 1;
//					}else{
//						++amountOfNeededLines;
//					}
//				}
//				if(amountOfNeededLines < amountOfLinesPerThread - 2){
//					std::list<std::string>::const_iterator itLine = (*it)->m_lines.begin();
//					for(int i = 0; i <= (int) (*it)->m_lines.size() - amountOfLinesPerThread + 2; ++i){
//						++itLine; // jump over elements in the list which are no needed at the moment
//					}
//					for(; itLine != (*it)->m_lines.end(); ++itLine, ++counter){
//						if(itLine->length() < colWidth){
//							mvprintw(row * amountOfLinesPerThread + counter + actLine, isLeft ? 4 : startOfRight + 1, itLine->c_str());
//						}else{
//							std::string line = *itLine;
//							int diff = 0;
//							while(line.length() > colWidth){
//								mvprintw(row * amountOfLinesPerThread + counter + actLine, isLeft ? 4 + diff: startOfRight + 1 + diff, line.substr(0, colWidth - diff).c_str());
//								++counter;
//								line = line.substr(colWidth, line.length() - colWidth);
//								diff = 2;
//							}
//							mvprintw(row * amountOfLinesPerThread + counter + actLine, isLeft ? 4 + diff: startOfRight + 1 + diff, line.substr(0, colWidth - diff).c_str());
//						}
//					}
//				}else{
//					counter += 2;
//					for(std::list<std::string>::reverse_iterator itLine = (*it)->m_lines.rbegin(); itLine != (*it)->m_lines.rend(); ++itLine, ++counter){
//						if(itLine->length() < colWidth){
//							if(counter < amountOfLinesPerThread){ // -1 for the start line
//								mvprintw((row + 1) * amountOfLinesPerThread - counter + actLine, isLeft ? 4 : startOfRight + 1, itLine->c_str());
//							}
//						}else{
//							std::string line = *itLine;
//							counter += ceil((itLine->length() - colWidth) / (double) (colWidth - 2));
//							int diff = 0;
//							while(line.length() > colWidth){
//								if(counter < amountOfLinesPerThread){ // -1 for the start line
//									mvprintw((row + 1) * amountOfLinesPerThread - counter + actLine, isLeft ? 4 + diff : startOfRight + 1 + diff, line.substr(0, colWidth - diff).c_str());
//								}
//								diff = 2;
//								line = line.substr(colWidth, line.length() - colWidth);
//								--counter;
//							}
//							if(counter < amountOfLinesPerThread){ // -1 for the start line
//								mvprintw((row + 1) * amountOfLinesPerThread - counter + actLine, isLeft ? 4  + diff : startOfRight + 1  + diff, line.substr(0, colWidth - diff).c_str());
//							}
//							counter += ceil((itLine->length() - colWidth) / (double) (colWidth - 2));
//						}
//					}
//				}
//				attron(COLOR_PAIR(1));
//				const int diff = (*it)->m_lines.size() - heightOfThreadInfo;
//				for(int i = 0; i < diff; ++i){
//					(*it)->m_lines.pop_front(); // to reduce it to HEIGHT OF THREAD INFO is the maximal which can be displayed so the rest can be erased
//				}
//			}
	(*it)->m_lineMutex.unlock();
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
