/*
 * UsefulStuff.h
 *
 *  Created on: 31.05.2016
 *      Author: Max
 */

#ifndef UTILITY_UTIL_H_
#define UTILITY_UTIL_H_


#include <iostream>

template<typename T>
inline std::string number2String(const T& in){
	std::stringstream ss; ss << in;
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
	std::cout << "Error in " << __PRETTY_FUNCTION__ << ":" << number2String(__LINE__) << ": " << message << std::endl; \




#endif /* UTILITY_UTIL_H_ */
