/*
 * UsefulStuff.h
 *
 *  Created on: 31.05.2016
 *      Author: Max
 */

#ifndef UTILITY_UTIL_H_
#define UTILITY_UTIL_H_

#include <iostream>
#include <sstream>
#include <fstream>
#include <istream>
#include <iostream>
#include <string>
#include <vector>
#include "StopWatch.h"

#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

template<typename T>
inline std::string number2String(const T& in){
	std::stringstream ss;
	ss << in;
	return ss.str();
}

#define printMsg(message, ...)\
	do{ \
		const std::string p[] = {__VA_ARGS__}; \
		const int numArgs = sizeof(p)/sizeof(p[0]); \
		std::string resultString = numArgs>0 ? "" : " "; \
		for(int i = 0; i < numArgs; ++i)\
			resultString += p[i]; \
		std::cout << resultString << __PRETTY_FUNCTION__ << ":" << number2String(__LINE__) << ": " << message << std::endl; \
	}while(0); \

#define printError(message) \
	std::cout << RED << "Error in " << __PRETTY_FUNCTION__ << ":" << number2String(__LINE__) << ": " << RESET << message << std::endl \

#define printWarning(message) \
	std::cout << YELLOW << "Warning in " << __PRETTY_FUNCTION__ << ":" << number2String(__LINE__) << ": " << RESET << message << std::endl \

#define printLine() \
		std::cout << "Debug in " << __PRETTY_FUNCTION__ << ":" << number2String(__LINE__) << std::endl \

template<class T> const T& min(const T& a, const T& b){
	return !(b < a) ? a : b;     // or: return !comp(b,a)?a:b; for version (2)
}

template<class T> const T& max(const T& a, const T& b){
	return !(b > a) ? a : b;     // or: return !comp(b,a)?a:b; for version (2)
}

namespace Utility {

// reading and writing of binary


}

#endif /* UTILITY_UTIL_H_ */
