/*
 * ThreadSafeOutput.h
 *
 *  Created on: 12.07.2016
 *      Author: Max
 */

#ifndef UTILITY_THREADSAFEOUTPUT_H_
#define UTILITY_THREADSAFEOUTPUT_H_

#include "Util.h"
#include <boost/thread.hpp> // Boost threads

class ThreadSafeOutput {
public:
	ThreadSafeOutput(std::ostream& stream = std::cout);
	virtual ~ThreadSafeOutput();

	void print(const std::string& text);

	void print(const std::string& text, const char* color);

private:
	boost::mutex m_mutex;
	std::ostream& m_stream;
};

#endif /* UTILITY_THREADSAFEOUTPUT_H_ */
