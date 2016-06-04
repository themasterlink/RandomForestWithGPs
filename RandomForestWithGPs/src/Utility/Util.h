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
#include <string>
#include <vector>
#include "StopWatch.h"

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
	std::cout << "Error in " << __PRETTY_FUNCTION__ << ":" << number2String(__LINE__) << ": " << message << std::endl \

#define printWarning(message) \
	std::cout << "Warning in " << __PRETTY_FUNCTION__ << ":" << number2String(__LINE__) << ": " << message << std::endl \

#define printLine() \
		std::cout << "Debug in " << __PRETTY_FUNCTION__ << ":" << number2String(__LINE__) << std::endl \

template<class T> const T& min(const T& a, const T& b){
	return !(b < a) ? a : b;     // or: return !comp(b,a)?a:b; for version (2)
}

template<class T> const T& max(const T& a, const T& b){
	return !(b > a) ? a : b;     // or: return !comp(b,a)?a:b; for version (2)
}

#endif /* UTILITY_UTIL_H_ */
