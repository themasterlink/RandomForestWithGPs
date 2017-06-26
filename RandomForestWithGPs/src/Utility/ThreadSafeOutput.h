/*
 * ThreadSafeOutput.h
 *
 *  Created on: 12.07.2016
 *      Author: Max
 */

#ifndef UTILITY_THREADSAFEOUTPUT_H_
#define UTILITY_THREADSAFEOUTPUT_H_

#include "Util.h"

class ThreadSafeOutput {
public:
	ThreadSafeOutput(std::ostream& stream = std::cout);
	virtual ~ThreadSafeOutput();

	void print(const std::string& text);

	void printInColor(const std::string& text, const char* color);

	void printSwitchingColor(const std::string& text);

private:
	Mutex m_mutex;
	std::ostream& m_stream;
	bool m_change;
};

#endif /* UTILITY_THREADSAFEOUTPUT_H_ */
